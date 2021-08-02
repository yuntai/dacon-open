import os
import argparse
import random
import numpy as np
from typing import Optional

from numpy.random import default_rng
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pytorch_lightning as pl

from transformers import BertForMaskedLM, AutoTokenizer, AdamW
from torch.utils.data.sampler import WeightedRandomSampler

from common import F1_Loss
from sampler import DistributedSamplerWrapper

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = BertForMaskedLM.from_pretrained(args.base_model, cache_dir=args.cache_dir)

        config = self.model.config
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_classes)

        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()

        self.loss_fct = F1_Loss(args.num_classes)

        self.val_f1 = torchmetrics.F1(num_classes=args.num_classes, average='macro')

    def forward(self, batch):
        return self.__step(batch)

    def __step(self, batch):
        cls_labels = batch.pop('cls_labels')
        out = self.model(**batch, output_hidden_states=True)
        lm_loss = out['loss']
        last_hidden_state = out['hidden_states'][-1]
        pooled_hidden_state = last_hidden_state.mean(dim=1)
        x = self.activation(pooled_hidden_state)
        x = self.dropout(x)
        logits = self.classifier(x)
        #cls_loss = F.cross_entropy(logits, cls_labels)
        cls_loss = self.loss_fct(logits, cls_labels)
        return {
            'lm_loss': lm_loss,
            'cls_loss': cls_loss,
            'last_hidden_state': last_hidden_state,
            'logits': logits,
            'cls_labels': cls_labels
        }

    def training_step(self, batch, batch_idx):
        out = self.__step(batch)
        loss = out['lm_loss'] + out['cls_loss']
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.__step(batch)
        loss = out['lm_loss'] + out['cls_loss']

        cls_labels = out['cls_labels']
        #preds = logits.argmax(dim=1)
        self.val_f1(out['logits'], cls_labels)
        __kwargs = dict(on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, **__kwargs)
        self.log('val_loss', loss, **__kwargs)

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            mode="min",
            patience=3,
        )
        checkpoint = ModelCheckpoint(
            dirpath=f"./res/mlm_base={self.hparams.base_model}&max_seq_len={self.hparams.max_seq_len}",
            monitor="val_loss",
            save_top_k=3,
            filename='{epoch}-{step}-{val_loss:.3f}-{val_f1:.3f}',
            mode='min'
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

        ___sz = 174304
        num_batches = ___sz // self.hparams.batch_size
        num_training_steps = num_batches * self.hparams.max_epochs
        num_warmup_steps = 0

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        lr_scheduler = LambdaLR(optimizer, lr_lambda, -1)
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
        from common import with_cache, get_weights
        tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)
        df = with_cache(prep_txt, self.prep_cache_path)(tokenizer)

        tr_df = df[df.cv != 0]
        va_df = df[df.cv == 0]

        tr_df, weights = get_weights(tr_df)
        self.weights = weights

        self.tr_ds = OpenMLMDataset(tr_df, tokenizer)
        self.va_ds = OpenMLMDataset(va_df, tokenizer)

    def train_dataloader(self):
        weighted_sampler = WeightedRandomSampler(self.weights, len(self.weights))
        return DataLoader(
            self.tr_ds,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=DistributedSamplerWrapper(weighted_sampler),
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
    def __init__(self, df, tokenizer, max_seq_len=512, add_speical_tok=False, seed=42):
        super().__init__()
        self.df = df.copy()
        self.max_seq_len = max_seq_len
        self.add_speical_tok = add_speical_tok
        self.tokenizer = tokenizer
        self.seed = seed
        self.mask_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index(tokenizer.special_tokens_map['mask_token'])]
        self.rng = default_rng(seed)

    def random_words(self, toks):
        probs = self.rng.random(len(toks))
        lm_label = toks.copy()
        ix = probs < 0.15
        toks[ix] = self.mask_id
        lm_label[~ix] = -100
        return toks, lm_label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        self.rng.shuffle(row.ixs)

        toks = row.toks[row.ixs]
        tok_lens = row.tok_lens[row.ixs]
        toks = np.concatenate(toks).astype(int)
        label = row.label

        if toks.size > self.max_seq_len:
            ovf = self.max_seq_len - toks.size
            if self.rng.random(1).item() < 0.5:
                toks = toks[:ovf]
            else:
                toks = toks[-ovf:]

        toks, lm_label = self.random_words(toks)
        toks = torch.tensor(toks)
        lm_label = torch.tensor(lm_label)

        sz = toks.size(0)
        attention_mask = torch.zeros(self.max_seq_len)
        attention_mask[:sz] = 1

        pad_sz = self.max_seq_len-sz
        toks = F.pad(toks, (0, pad_sz))
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
    from common import prep_cv

    df = pd.read_csv("/mnt/datasets/open/train.csv")
    df = prep_cv(df)

    kwargs = {
        'add_special_tokens': False,
        'padding': False,
        'return_attention_mask': False,
        'truncation': False,
        'return_token_type_ids': False,
        'return_tensors': None,
    }

    cols = ['과제명', '요약문_연구목표', '요약문_연구내용', '요약문_기대효과']
    df[cols] = df[cols].fillna('')

    toks_lst = []
    len_lst = []
    for t in tqdm(df[cols].itertuples()):
        t = t[1:]
        sents = sum([t[i].split('\n') for i in range(len(cols))], [])
        sents = [s.strip() for s in sents if len(s.strip()) > 0]
        toks = tokenizer(sents, **kwargs)['input_ids']
        toks_lst.append(np.array(toks,dtype=object))
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

def fit():
    from common import get_default_parser
    parser = get_default_parser()
    args = parser.parse_args()
    dm = OpenDataModule(args)
    model = LitMLModel(args)
    trainer = pl.Trainer(
        gpus=args.gpus,
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        max_epochs=args.max_epochs
    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    fit()
