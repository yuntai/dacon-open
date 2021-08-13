import os
import argparse
import random
import logging
import itertools
import numpy as np
from typing import Optional

from sklearn.metrics import classification_report
from numpy.random import default_rng
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import RobertaForMaskedLM

import pytorch_lightning as pl

from transformers import AutoTokenizer, AdamW, AutoModel

from sampler import DistributedSamplerWrapper

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='The initial learning rate for Adam.')
    parser.add_argument('--base_model', type=str, default='xlm-roberta-base')
    parser.add_argument('--cache_dir', type=str, default="./.cache")
    parser.add_argument('--dataroot', type=str, default="./datasets")
    parser.add_argument('--cv_size', type=int, default=5)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_classes', type=int, default=46)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=300)

    # wandb related
    parser.add_argument('--project', type=str, default='dacon-open-mlm')
    return parser

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class LitMLModel(pl.LightningModule):
    @staticmethod
    def extract_feature(ckpt_path):
        from common import prep, with_cache, get_dataset
        from tqdm.auto import tqdm
        import pathlib
        model = LitMLModel.load_from_checkpoint(ckpt_path)
        model.eval().cuda().freeze()
        args = model.hparams
        args.use_keywords = True
        args.use_english = True
        args.dataroot = pathlib.Path("/mnt/datasets/open")

        df = with_cache(prep, './prep/mlm_feat.pkl')(args)

        ds = get_dataset(df)
        dl = DataLoader(ds, batch_size=32, shuffle=False)

        hidden_states = []
        with torch.no_grad():
            for batch in tqdm(dl):
                batch.pop('label')
                item = {k:v.to('cuda') for k, v in batch.items()}
                item['output_hidden_states'] = True
                out = model(item)
                hidden_states.append(out['hidden_states'][-1].mean(axis=1).cpu())

        hs = torch.cat(hidden_states)
        hs = list(map(lambda x: torch.squeeze(x).numpy(), torch.split(hs, 1)))
        df['hs'] = hs

        hs_df = df.groupby('index')['hs'].apply(list)
        df.pop('hs')

        cols = ['index', 'label', 'cv']
        df = df.groupby('index')[cols].first().set_index('index').merge(hs_df, left_index=True, right_index=True, how='inner')
        df['seq_len'] = df.hs.apply(len)
        df['hs'] = df['hs'].apply(torch.tensor)

        return df

    def __init__(self, args, class_weight=None):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = RobertaForMaskedLM.from_pretrained(self.hparams.base_model, cache_dir=self.hparams.cache_dir)

        config = self.model.config
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.hparams.num_classes)

        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()

        #self.loss_fct = F1_Loss(args.num_classes)
        self.loss_fct = nn.CrossEntropyLoss(weight=class_weight)

        self.val_f1 = torchmetrics.F1(num_classes=self.hparams.num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(num_classes=self.hparams.num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(num_classes=self.hparams.num_classes, average='macro')
        self.do_lm_loss = False

    def forward(self, batch):
        return self.__step(batch)

    def __step(self, batch):
        cls_labels = batch.pop('cls_labels')
        out = self.model(**batch, output_hidden_states=True)
        last_hidden_state = out['hidden_states'][-1]

        pooled_hidden_state1 = last_hidden_state.mean(dim=1)
        pooled_hidden_state2 = last_hidden_state.max(dim=1)[0]
        pooled_hidden_state = torch.cat([pooled_hidden_state1, pooled_hidden_state2], dim=-1)

        x = self.activation(pooled_hidden_state)
        x = self.dropout(x)
        logits = self.classifier(x)
        cls_loss = self.loss_fct(logits, cls_labels)

        ret =  {
            'cls_loss': cls_loss,
            'last_hidden_state': last_hidden_state,
            'logits': logits,
            'cls_labels': cls_labels
        }
        if self.do_lm_loss:
            ret['lm_loss'] = out['loss']
        return ret

    def training_step(self, batch, batch_idx):
        out = self.__step(batch)
        loss = out['cls_loss']
        if self.do_lm_loss:
            loss += out['lm_loss']
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.__step(batch)

        loss = out['cls_loss']
        if self.do_lm_loss:
            loss += out['lm_loss']

        cls_labels = out['cls_labels']
        self.val_f1(out['logits'], cls_labels)
        __kwargs = dict(on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, **__kwargs)
        self.log('val_loss', loss, **__kwargs)
        if self.do_lm_loss:
            self.log('val_lm_loss', out['lm_loss'], **__kwargs)
            self.log('val_cls_loss', out['cls_loss'], **__kwargs)

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val_f1",
            min_delta=0.00,
            mode="max",
            patience=5,
        )
        exp_name = self.trainer.logger.experiment.name
        logger.info(f"{exp_name=}")
        checkpoint = ModelCheckpoint(
            dirpath=f"./res/mlm_base={self.hparams.base_model}&max_seq_len={self.hparams.max_seq_len}/{exp_name}",
            monitor="val_f1",
            save_top_k=3,
            filename='{epoch}-{step}-{val_loss:.3f}-{val_f1:.3f}',
            mode='max'
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')

        return [early_stop, checkpoint, lr_monitor]

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{
            "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        ___sz = len(self.train_dataloader().dataset)
        num_batches = ___sz // self.hparams.batch_size
        num_training_steps = num_batches * self.hparams.max_epochs
        num_warmup_steps = self.hparams.warmup_steps

        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


class OpenDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.args = args
        self.prep_cache_path = f'./prep/{self.args.base_model}_mlm_prep.pkl'

    def prepare_data(self):
        from common import with_cache
        tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)
        with_cache(prep_txt, self.prep_cache_path)(tokenizer)

    def setup(self, stage: Optional[str] = None):
        from common import with_cache, get_class_weights, cv_split
        tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)
        df = with_cache(prep_txt, self.prep_cache_path)(tokenizer)

        tr_df, va_df = cv_split(df, self.args.cv)

        weights = get_class_weights(tr_df)
        self.weights = weights

        self.tr_ds = OpenMLMDataset(tr_df, tokenizer, self.args.max_seq_len, is_training=False)
        self.va_ds = OpenMLMDataset(va_df, tokenizer, self.args.max_seq_len, is_training=False)

    def train_dataloader(self):
        return DataLoader(
            self.tr_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=16)

    def val_dataloader(self):
        return DataLoader(
            self.va_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16)

    #def test_dataloader(self):
    #    return DataLoader(self.ds, batch_size=self.hparams.batch_size)

class OpenMLMDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_seq_len, add_speical_tok=False, seed=42, is_training=True):
        super().__init__()
        self.df = df.copy()
        self.max_seq_len = max_seq_len
        self.add_speical_tok = add_speical_tok
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index(tokenizer.special_tokens_map['mask_token'])]
        self.seed = seed
        self.rng = default_rng(seed)
        self.is_training = is_training

    def random_mask(self, toks):
        probs = self.rng.random(len(toks))
        lm_label = toks.copy()
        ix = probs < 0.15
        toks[ix] = self.mask_id
        lm_label[~ix] = -100
        return toks, lm_label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        #if self.is_training:
        self.rng.shuffle(row.ixs)

        toks = row.toks[row.ixs]
        tok_lens = row.tok_lens[row.ixs]
        toks = np.concatenate(toks).astype(int)
        label = row.label

        if toks.size > self.max_seq_len:
            if self.rng.random(1).item() < 0.5:
                toks = toks[:self.max_seq_len]
            else:
                toks = toks[-self.max_seq_len:]

        if self.is_training:
            toks, lm_label = self.random_mask(toks)
        else:
            lm_label = toks.copy()

        toks = torch.tensor(toks)
        lm_label = torch.tensor(lm_label)

        sz = toks.size(0)
        attention_mask = torch.zeros(self.max_seq_len)
        attention_mask[:sz] = 1

        pad_sz = self.max_seq_len - sz
        toks = F.pad(toks, (0, pad_sz), value=0)
        lm_label = F.pad(lm_label, (0, pad_sz), value=-100)

        return {
            'input_ids': toks,
            'attention_mask': attention_mask,
            'cls_labels': label,
            'labels': lm_label
        }

    def __len__(self):
        return self.df.shape[0]

def prep_txt(tokenizer):
    from tqdm.auto import tqdm
    from common import prep_cv_strat, clean_text, clean_records

    df = pd.read_csv("/mnt/datasets/open/train.csv")
    df_c = clean_records(df.copy())
    df = df[df.index.isin(df_c.index)]
    df = prep_cv_strat(df)

    kwargs = {
        'add_special_tokens': False,
        'padding': False,
        'return_attention_mask': False,
        'truncation': False,
        'return_token_type_ids': False,
        'return_tensors': None,
    }

    cols = ['과제명', '요약문_연구목표', '요약문_연구내용', '요약문_기대효과', '요약문_한글키워드', '요약문_영문키워드']
    df[cols] = df[cols].fillna('')

    toks_lst = []
    len_lst = []
    for t in tqdm(df[cols].itertuples()):
        t = t[1:]
        sents = sum([t[i].split('\n') for i in range(len(cols))], [])
        sents = [s.strip() for s in sents if len(s.strip()) > 0]
        sents = [clean_text(s) for s in sents]
        sents = [s.strip() for s in sents if len(s.strip()) > 0]
        toks = tokenizer(sents, **kwargs)['input_ids']
        toks_lst.append(np.array(toks, dtype=object))
        len_lst.append(np.array(list(map(len, toks))))

    df['toks'] = toks_lst
    df['tok_lens'] = len_lst
    df['tot_len'] = df['tok_lens'].apply(sum)
    df['ixs'] = df['toks'].apply(lambda x: np.arange(len(x)))

    cols = ['index', 'toks', 'tok_lens', 'tot_len', 'ixs', 'cv']
    if 'label' in df.columns:
        cols += ['label']

    df = df[cols]

    return df

def validate(ckpt):
    model = LitMLModel.load_from_checkpoint(ckpt)
    dm = OpenDataModule(argparse.Namespace(**model.hparams))
    trainer = pl.Trainer(gpus=1)
    trainer.validate(model, dm)

def fit():
    parser = get_default_parser()
    args = parser.parse_args()

    dm = OpenDataModule(args)
    model = LitMLModel(args)

    trainer = pl.Trainer(
        gpus=args.gpus,
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        max_epochs=args.max_epochs,
        logger=WandbLogger(project=args.project),
        replace_sampler_ddp=True,
        val_check_interval=0.5,
    )
    trainer.fit(model, dm)

def predict(ckpt):
    model = LitMLModel.load_from_checkpoint(ckpt)
    dm = OpenDataModule(args)
    w = torch.FloatTensor(dm.weights)
    model = LitMLModel(args, weights=w)
    trainer.predict(model, dm)
    p = trainer.predict(model, dm.val_dataloader(), ckpt_path=ckpt_path, return_predictions=True)
    logits, labels = zip(*map(itertools.itemgetter('logits', 'cls_labels'), p))
    logits = torch.cat(logits).cpu()
    labels = torch.cat(labels).cpu()
    preds = logits.argmax(dim=-1)
    res = classification_report(labels, preds)

    df = pd.DataFrame()
    for i in range(46):
        df = df.append(res[str(i)])

    return df

if __name__ == '__main__':
    fit()
