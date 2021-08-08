import random
import logging
import argparse
import re
import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn

import pandas as pd
import numpy as np

import torchmetrics
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import classification_report


from pytorch_lightning.loggers import WandbLogger

#from transformers import BertForSequenceClassification
from transformers import EarlyStoppingCallback
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler, AutoModel
from transformers import AdamW
from tqdm.auto import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import functools

import common

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

    parser.add_argument('--num_classes', type=int, default=46)

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
    parser.add_argument('--use_english', dest='use_english', action='store_true')
    parser.add_argument('--no_use_english', dest='use_english', action='store_false')
    parser.set_defaults(use_english=True)

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
        from common import prep, cv_split, with_cache
        is_test = test_cache_path is not None
        m, args = LitBaseModel.load_from_ckpt(ckpt_path)
        m.eval().cuda().freeze()

        if test_cache_path is None:
            df = with_cache(prep, m.cache_path)(args)
        else:
            __prep = functools.partial(prep, is_test=True)
            df = with_cache(__prep, test_cache_path)(args)

        ds = get_dataset(df, is_test=is_test)
        dl = DataLoader(ds, batch_size=128, shuffle=False)

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
        from common import prep, cv_split, with_cache
        m = LitBaseModel.load_from_checkpoint(ckpt_path)
        m.eval().cuda().freeze()

        # TODO: w/ trainer
        df = with_cache(prep, m.hparams.cache_path)(args)
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
            tmpl = './prep/seed={seed}&max_seq_len={max_seq_len}&base_model={base_model}&use_keywords={use_keywords}&use_english={use_english}.pkl'
            kwargs['cache_path'] = tmpl.format(**kwargs)

        self.save_hyperparameters(kwargs)

        self.base_model = BaseClassifier(self.hparams)

        __kwargs = {'num_classes': self.hparams.num_classes, 'average':'macro'}

        self.val_f1 = torchmetrics.F1(**__kwargs)
        self.val_acc = torchmetrics.Accuracy(**__kwargs)
        self.val_recall = torchmetrics.Recall(**__kwargs)
        self.val_precision = torchmetrics.Precision(**__kwargs)

        self._datasets = None

    def forward(self, batch):
        if 'labels' in batch:
            _ = batch.pop("labels")
        return self.base_model(**batch)

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits = self.base_model(**batch)['logits']
        loss = F.cross_entropy(logits, labels)
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
        loss = F.cross_entropy(logits, labels)
        return loss

    def train_dataloader(self) -> DataLoader:
        from sampler import DynamicBalanceClassSampler, DistributedSamplerWrapper
        ds = self.datasets['train']
        #train_labels = ds.targets.cpu().numpy().tolist()
        #train_sampler = DynamicBalanceClassSampler(train_labels, exp_lambda=0.95)
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            #sampler=DistributedSamplerWrapper(train_sampler),
            pin_memory=True,
            num_workers=16)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets['val'], batch_size=self.hparams.batch_size, num_workers=16)

    @property
    def datasets(self):
        if self._datasets:
            return self._datasets

        from common import prep, cv_split, get_weights, with_cache

        df = with_cache(prep, self.hparams.cache_path)(self.hparams)

        tr_df, va_df = cv_split(df, self.hparams.cv)
        tr_df, weights = get_weights(tr_df)

        tr_ds = get_dataset(tr_df)
        va_ds = get_dataset(va_df)

        self._datasets = {'train': tr_ds, 'val': va_ds, 'weights': weights}

        return self._datasets

    def prepare_data(self):
        from common import prep, with_cache
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
        num_warmup_steps = 0

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps-current_step) / float(max(1, num_training_steps-num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, -1)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val_f1",
            min_delta=0.00,
            mode="max",
            patience=3,
        )

        name = self.trainer.logger.experiment.name
        logger.info(f"exp {name=}")
        checkpoint = ModelCheckpoint(
            dirpath=f"./res/base_model={self.hparams.base_model}&max_seq_len={self.hparams.max_seq_len}/{name}",
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

if __name__ == '__main__' and not common.isin_ipython():
    parser = get_parser()
    args = parser.parse_args()
    args.dataroot = Path(args.dataroot)

    pl.seed_everything(args.seed)

    train_base_model(args)

