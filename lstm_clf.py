import random
import argparse
import re
from pathlib import Path

import torch
import torch.nn as nn

import pandas as pd
import numpy as np

import torchmetrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger

from transformers import BertTokenizerFast, BertModel, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from transformers import AdamW
from tqdm.auto import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from common import cv_split, with_cache, get_weights
from base_model import LitBaseModel

from sklearn.metrics import classification_report
from  torch.nn.utils.rnn import pack_padded_sequence

def get_parser():

    parser = argparse.ArgumentParser()
    #parser.add_argument('--max_seq_len', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    #CKPT = './res/base_model=xlm-roberta-base&max_seq_len=250/epoch=3-step=18767-val_loss=0.337-val_f1=0.673.ckpt'
    CKPT = "./res/bertbase_max_len=250/epoch=16-val_f1=0.74-val_loss=0.59.ckpt"
    parser.add_argument('--ckpt', type=str, default=CKPT)

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='The initial learning rate for Adam.')
    MODELS = [ 'bert-base-multilingual-cased', 'xlm-roberta-base', 'xlm-roberta-large', 'monologg/kobert', 'monologg/distilkobert']
    #parser.add_argument('--base_model', choices=MODELS, default='bert-base-multilingual-cased')
    parser.add_argument('--cache_dir', type=str, default="./.cache", help="huggingface cahce dir")
    parser.add_argument('--dataroot', type=str, default="/mnt/datasets/open")
    # wandb related
    parser.add_argument('--project', type=str, default='dacon-open')
    parser.add_argument('--name', type=str, default=None)

    return parser

class OpenSeqDataset(Dataset):
    def __init__(self, df, dim=768, labels=None):
        self.df = df
        #TODO: didnt need to convett to numpy from the beg
        max_seq_len = self.df['seq_len'].max()
        self.labels = labels
        self.zeros = torch.zeros(max_seq_len, dim)

    def __getitem__(self, idx):
        hs, seq_len = self.df.iloc[idx][['hs','seq_len']]
        hs = torch.cat([hs, self.zeros[seq_len:]])
        ret = (hs, seq_len)
        if self.labels:
            ret += (self.labels[idx],)
        return ret

    def __len__(self):
        return self.df.shape[0]

class LitOpenSeq(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitOpenSeq")
        parser.add_argument('--hidden_size', type=int, default=1024)
        parser.add_argument('--input_size', type=int, default=768)
        parser.add_argument('--dropout_prob', type=float, default=0.1)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--initializer_range', type=float, default=0.02)
        return parent_parser

    def __step(self, x, seq_len):
        packed_input = pack_padded_sequence(x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.lstm(packed_input)

        x = self.activation(ht[-1])
        x = self.dropout(x)
        x = self.lin(x)
        logits = self.clf(x)

        return logits

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self._datasets = None

        self.num_classes = 2
        kwargs = {'num_classes': self.num_classes, 'average':'macro'}
        self.val_f1 = torchmetrics.F1(**kwargs)
        self.val_acc = torchmetrics.Accuracy(**kwargs)
        self.val_recall = torchmetrics.Recall(**kwargs)
        self.val_precision = torchmetrics.Precision(**kwargs)
        self.cache_path = "./prep/seq.pkl"

        self.lstm = nn.LSTM(
            input_size=args.input_size, # dim
            hidden_size=args.hidden_size,
            batch_first=True,
            num_layers=args.num_layers,
            bidirectional=True)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout_prob)
        self.lin = nn.Linear(args.hidden_size, 512)
        self.clf = nn.Linear(512, self.num_classes)

        self._init_weights(self.lin)
        self._init_weights(self.clf)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def prepare_data(self):
        with_cache(LitBaseModel.extract_feature, self.cache_path)(self.hparams.ckpt)

    def load_datasets(self):
        df = with_cache(LitBaseModel.extract_feature, self.cache_path)(self.hparams.ckpt)
        tr_df, va_df = cv_split(df, self.hparams.cv)
        tr_df.loc[tr_df.label > 0, 'label'] = 1
        va_df.loc[va_df.label > 0, 'label'] = 1
        tr_df = get_weights(tr_df)
        weights = tr_df.pop('w').values.tolist()

        tr_ds = OpenSeqDataset(tr_df, labels=tr_df['label'].values.tolist())
        va_ds = OpenSeqDataset(va_df, labels=va_df['label'].values.tolist())
        return {'train': tr_ds, 'val': va_ds, 'weights': weights}

    def train_dataloader(self) -> DataLoader:
        from sampler import DistributedSampler

        ds = self.get_datasets()
        w = ds['weights']
        weighted_sampler = WeightedRandomSampler(w, len(w))

        return DataLoader(
            ds['train'],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=DistributedSampler(weighted_sampler),
            pin_memory=True,
            num_workers=16)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.get_datasets()['val'], batch_size=self.hparams.batch_size, num_workers=16)

    def get_datasets(self):
        if self._datasets is None:
            self._datasets = self.load_datasets()

        return self._datasets

    def forward(self, batch):
        x, seq_len, _ = batch
        logits = self.__step(x)
        return logits.argmax(dim=1)

    def validation_step(self, batch, batch_idx):
        x, seq_len, labels = batch
        logits = self.__step(x, seq_len)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)

        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)

        kwargs = dict(on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, **kwargs)
        self.log('val_f1', self.val_f1, **kwargs)
        self.log('val_precision', self.val_precision, **kwargs)
        self.log('val_recall', self.val_recall, **kwargs)
        self.log('val_loss', loss, **kwargs)

    def training_step(self, batch, batch_idx):
        x, seq_len, labels = batch
        logits = self.__step(x, seq_len)
        loss = F.cross_entropy(logits, labels)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{
            "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(optimizer),
            "monitor": "val_f1"
        }

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode='min'
        )
        early_stop_callback_f1 = EarlyStopping(
            monitor='val_f1',
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode='max'
        )
        mc = ModelCheckpoint(
            dirpath=f"./res/lstm_seq",
            monitor="val_loss",
            save_top_k=3,
            filename='{epoch}-{step}-{val_loss:.3f}-{val_f1:.3f}',
            mode='min'
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return [early_stop_callback_f1, mc, lr_monitor]

def train_lstm_classifier(args):
    model = LitOpenSeq(args)

    wandb_logger = WandbLogger(
        project=args.project,
        name=args.name
    )
    trainer = pl.Trainer(
        gpus=args.gpus,
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        max_epochs=args.max_epochs,
        checkpoint_callback=False,
        logger=wandb_logger,
        replace_sampler_ddp=False,
        val_check_interval=0.5,
    )
    trainer.fit(model)

if __name__ == '__main__':
    parser = get_parser()
    parser = LitOpenSeq.add_model_specific_args(parser)
    #parser = pl.Trainer.add_argparse_args(parser) 
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    train_lstm_classifier(args)
