import random
import logging
import itertools
import operator
import re
import os
import argparse
import functools
from pathlib import Path
from typing import Optional
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torchmetrics

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from transformers import AutoModel, AdamW, EarlyStoppingCallback, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

def isin_ipython():
    try:
        return get_ipython().__class__.__name__ == 'TerminalInteractiveShell'
    except NameError:
        return False

def with_cache(func, cache_path):
    def __inner(*args, **kwargs):
        if Path(cache_path).exists():
            print(f"CACHE FOUND {cache_path}")
            df = pd.read_pickle(cache_path)
        else:
            print(f"CACHE NOT FOUND {cache_path}")
            df = func(*args, **kwargs)
            print(f"saving to {cache_path} ...")
            df.to_pickle(cache_path)
        return df
    return __inner

def clean_text(s):
    s = re.sub("(\\W)+"," ", s)
    return s.strip()

def get_split(text1, chunk_size=250, overlap_pct=0.25):
    words = text1.split()
    overlap_sz = int(overlap_pct * chunk_size)

    step_size = chunk_size - overlap_sz

    res = [words[:chunk_size]]
    if len(words) > chunk_size:
        res += [words[i:i+chunk_size] for i in range(step_size, len(words), step_size)]

    return list(map(lambda x: " ".join(x), res))

def prep_explode(df, max_seq_len, is_test=False):
    __get_split = functools.partial(get_split, chunk_size=max_seq_len)
    df['data_split'] = df['data_cleaned'].apply(__get_split)

    train_l, label_l, cv_l, index_l, subix_l = [], [], [], [], []

    for _, row in df.iterrows():
      for ix, l in enumerate(row['data_split']):
          train_l.append(l)
          if not is_test:
            label_l.append(row['label'])
            cv_l.append(row['cv'])
          index_l.append(row['index'])
          subix_l.append(ix)

    data = {'index': index_l, 'subix': subix_l, 'data': train_l}
    if not is_test:
        data['label'] = label_l
        data['cv'] = cv_l
    return pd.DataFrame(data)

def prep_tok(df, tokenizer, add_special_tokens=False):
    TOK_COLS = ['input_ids', 'attention_mask']
    kwargs = {
        'add_special_tokens': add_special_tokens,
        'padding': True,
        'return_attention_mask': True,
        'truncation': True
    }
    toks = tokenizer(df['data'].values.tolist(), **kwargs)
    for k in TOK_COLS:
        if k in toks:
            df[k] = toks[k]

    return df

def prep_cv_strat(df, cv_size=5, seed=42):
    df['cv'] = -1
    skf = StratifiedKFold(n_splits=cv_size, shuffle=True, random_state=seed)
    for _cv, (_, test_index) in enumerate(skf.split(np.zeros(len(df.label)), df.label)):
        df.iloc[test_index, df.columns.get_loc('cv')] = _cv
    df['cv'] = df.cv.astype(int)
    return df

def prep_txt(df, include_keywords=True, is_test=False):
    cols = ['과제명', '요약문_연구목표', '요약문_연구내용', '요약문_기대효과']
    if include_keywords:
        cols += ['요약문_한글키워드', '요약문_영문키워드']

    if is_test:
        df = df[cols + ['index']].copy()
    else:
        df = df[cols + ['index', 'label']].copy()
    df.fillna(' ', inplace=True)

    df['data'] = df[cols[0]]
    for c in cols[1:]:
        df['data'] += ' ' + df[c]

    df['data_cleaned'] = df['data'].apply(clean_text)

    return df

def prep(args, is_test=False):
    fn = 'test.csv' if is_test else 'train.csv'
    df = pd.read_csv(args.dataroot/fn)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, do_lower_case=False)

    df = prep_txt(df, args.use_keywords, is_test=is_test)
    if not is_test:
        df = prep_cv_strat(df, cv_size=args.cv_size, seed=args.seed)
    df = prep_explode(df, args.max_seq_len, is_test=is_test)
    df = prep_tok(df, tokenizer)
    return df

def prep_cv_strat(df, cv_size=5, seed=42):
    df['cv'] = -1
    skf = StratifiedKFold(n_splits=cv_size, shuffle=True, random_state=seed)
    for _cv, (_, test_index) in enumerate(skf.split(np.zeros(len(df.label)), df.label)):
        df.iloc[test_index, df.columns.get_loc('cv')] = _cv
    df['cv'] = df.cv.astype(int)
    return df

def get_class_weights(df, col='label'):
    w = df.groupby(col)[col].count()
    w = w.sum()/w
    return w.values.tolist()

def cv_split(df, cv):
    tr_df = df.loc[df.cv!=cv]
    va_df = df.loc[df.cv==cv]
    return tr_df, va_df

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# create torch dataset
class OpenDataset(Dataset):
    def __init__(self, df, labels=None):
        self.df = df
        if labels:
            self.targets = torch.LongTensor(labels)
        else:
            self.targets = None

    def __getitem__(self, idx):
        #TODO: conversion to tensor necessario?
        item = {k: torch.tensor(v) for k, v in self.df.iloc[idx].to_dict().items()}
        if self.targets is not None:
            item["labels"] = self.targets[idx]
        return item

    def __len__(self):
        return self.df.shape[0]

def get_dataset(df, is_test=False):
    X = df[['input_ids', 'attention_mask']]
    y = df['label'].values.tolist() if not is_test else None
    return OpenDataset(X, y)

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_len', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_steps', type=int, default=300)

    #parser.add_argument('--num_classes', type=int, default=46)

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='The initial learning rate for Adam.')
    MODELS = [ 'bert-base-multilingual-cased', 'xlm-roberta-base', 'monologg/kobert', 'monologg/distilkobert']
    parser.add_argument('--base_model', choices=MODELS, default='xlm-roberta-base')
    parser.add_argument('--cache_dir', type=str, default="./.cache")
    parser.add_argument('--dataroot', type=str, default="/mnt/datasets/open")

    parser.add_argument('--use_keywords', dest='use_keywords', action='store_true')
    parser.add_argument('--no_use_keywords', dest='use_keywords', action='store_false')
    parser.set_defaults(use_keywords=True)

    parser.add_argument('--cv_size', type=int, default=5)

    # wandb related
    parser.add_argument('--project', type=str, default='dacon-open')

    return parser

class BaseClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.m = AutoModel.from_pretrained(args.base_model, cache_dir=args.cache_dir)
        config = self.m.config

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_classes)

        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        output = self.m(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden = output['last_hidden_state'].mean(axis=1)
        x = self.activation(hidden)
        x = self.dropout(x)

        logits = self.classifier(x)

        return {'logits': logits, 'hidden': hidden}

class LitBaseModel(pl.LightningModule):
    @staticmethod
    def extract_feature(ckpt_path, test_cache_path=None):
        is_test = test_cache_path is not None
        m = LitBaseModel.load_from_checkpoint(ckpt_path)
        args = m.hparams
        m.eval().cuda().freeze()

        if test_cache_path is None:
            df = with_cache(prep, m.hparams.cache_path)(args)
        else:
            __prep = functools.partial(prep, is_test=True)
            df = with_cache(__prep, test_cache_path)(args)

        ds = get_dataset(df, is_test=is_test)
        dl = DataLoader(ds, batch_size=128, shuffle=False)

        logging.info("extracting feature...")
        hidden_states = []
        with torch.no_grad():
            for batch in tqdm(dl):
                if not is_test:
                    batch.pop('labels')
                item = {k:v.to('cuda') for k, v in batch.items()}
                out = m(item)
                hidden_states.append(out['hidden'].cpu())

        hs = torch.cat(hidden_states)
        hs = list(map(lambda x: torch.squeeze(x).numpy(), torch.split(hs, 1)))
        df['hs'] = hs

        hs_df = df.groupby('index')['hs'].apply(list)
        df.pop('hs')

        cols = ['index']
        if not is_test:
            cols += ['label', 'cv']
        df = df.groupby('index')[cols].first().set_index('index').merge(hs_df, left_index=True, right_index=True, how='inner')
        df['seq_len'] = df.hs.apply(len)
        df['hs'] = df['hs'].apply(torch.tensor)

        return df

    @staticmethod
    def eval_from_ckpt(ckpt_path):
        m = LitBaseModel.load_from_checkpoint(ckpt_path)
        m.eval().cuda().freeze()

        # TODO: w/ trainer
        df = with_cache(prep, m.hparams.cache_path)(m.hparams)
        _, va_df = cv_split(df, m.hparams.cv)
        va_ds = get_dataset(va_df)
        va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

        preds, labels = [], []
        with torch.no_grad():
            for x in tqdm(va_loader):
                label = x.pop('labels')
                item = {k:v.to('cuda') for k, v in x.items()}
                logits = m(item)['logits']
                preds.append(logits.cpu().argmax(axis=-1))
                labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        print(classification_report(labels, preds))
        return (preds, labels)

    def __init__(self, **kwargs):
        super().__init__()
        if 'cache_path' not in kwargs:
            tmpl = './prep/seed={seed}&max_seq_len={max_seq_len}&base_model={base_model}&use_keywords={use_keywords}.pkl'
            kwargs['cache_path'] = tmpl.format(**kwargs)

        self.save_hyperparameters(kwargs)

        self.base_model = BaseClassifier(self.hparams)

        __kwargs = {'num_classes': self.hparams.num_classes, 'average':'macro'}

        self.loss_fct = None
        self.val_f1 = torchmetrics.F1(**__kwargs)
        self.val_acc = torchmetrics.Accuracy(**__kwargs)
        self.val_recall = torchmetrics.Recall(**__kwargs)
        self.val_precision = torchmetrics.Precision(**__kwargs)

        self._datasets = None

    def forward(self, batch):
        labels = None
        if 'labels' in batch:
            labels = batch.pop("labels")
        out = self.base_model(**batch)

        if labels is not None:
            out["labels"] = labels

        return out

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits = self.base_model(**batch)['logits']
        loss = self.loss_fct(logits, labels)
        preds = logits.argmax(dim=1)

        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)

        __kwargs = dict(on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, **__kwargs)
        self.log('val_f1', self.val_f1, **__kwargs)
        self.log('val_precision', self.val_precision, **__kwargs)
        self.log('val_recall', self.val_recall, **__kwargs)
        self.log('val_loss', loss, **__kwargs)

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits = self.base_model(**batch)['logits']
        loss = self.loss_fct(logits, labels)
        return loss

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['train'],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets['val'], batch_size=self.hparams.batch_size, num_workers=16)

    @property
    def datasets(self):
        assert self._datasets is not None
        return self._datasets

    def setup(self, stage: Optional[str] = None):
        df = with_cache(prep, self.hparams.cache_path)(self.hparams)

        tr_df, va_df = cv_split(df, self.hparams.cv)

        assert sorted(tr_df.label.unique().tolist()) == list(range(46))
        assert sorted(va_df.label.unique().tolist()) == list(range(46))

        tr_ds = get_dataset(tr_df)
        va_ds = get_dataset(va_df)

        self._datasets = {'train': tr_ds, 'val': va_ds}
        weight = get_class_weights(tr_df)

        self.loss_fct = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))

    def prepare_data(self):
        with_cache(prep, self.hparams.cache_path)(self.hparams)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{
            "params": [p for n, p in self.base_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in self.base_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        num_batches = len(self.datasets['train']) // self.hparams.batch_size
        num_training_steps = num_batches * self.hparams.max_epochs
        num_warmup_steps = self.hparams.warmup_steps

        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val_f1",
            min_delta=0.00,
            mode="max",
            patience=4,
        )

        exp_name = self.trainer.logger.experiment.name
        logger.info(f"{exp_name=}")
        checkpoint = ModelCheckpoint(
            dirpath=f"./res/base_model={self.hparams.base_model}&max_seq_len={self.hparams.max_seq_len}/{exp_name}",
            monitor="val_f1",
            save_top_k=3,
            filename='{epoch}-{step}-{val_loss:.3f}-{val_f1:.3f}',
            mode='max',
            every_n_epochs=1,
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')

        return [early_stop, checkpoint, lr_monitor]

def train_base_model(args):
    model = LitBaseModel(**args.__dict__)

    trainer = pl.Trainer(
        gpus=args.gpus,
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        max_epochs=args.max_epochs,
        checkpoint_callback=False,
        logger=WandbLogger(project=args.project),
        replace_sampler_ddp=True,
        val_check_interval=0.5,
    )
    trainer.fit(model)

def predict(ckpt_path):
    assert ckpt_path is not None or name is not None

    model = LitBaseModel.load_from_checkpoint(ckpt_path)
    trainer = pl.Trainer(gpus=1, amp_level='O2', precision=16)

    p = trainer.predict(dataloaders=model.val_dataloader(), return_predictions=True)
    logits, labels = zip(*map(operator.itemgetter('logits', 'labels'), p))
    logits = torch.cat(logits).cpu()
    labels = torch.cat(labels).cpu()
    preds = logits.argmax(dim=-1)
    res = classification_report(labels, preds, output_dict=True)

    df = pd.DataFrame()
    for i in range(46):
        df = df.append(res[str(i)], ignore_index=True)

    return {
        'report': df,
        'logits': logits,
        'labels': labels
    }

if __name__ == '__main__' and not isin_ipython():
    parser = get_parser()
    args = parser.parse_args()
    args.dataroot = Path(args.dataroot)
    args.num_classes = 46

    pl.seed_everything(args.seed)

    train_base_model(args)
