import random
import argparse
import re
import os
import operator
import pickle
import logging

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
from common import cv_split, with_cache
from base_model import LitBaseModel
from pytorch_lightning.trainer.states import TrainerFn

from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pack_padded_sequence
from common import F1_Loss

def isin_ipython():
    try:
        return get_ipython().__class__.__name__ == 'TerminalInteractiveShell'
    except NameError:
        return False      # Probably standard Python interpreter

def get_class_weights(df, col='label'):
    w = df.groupby(col)[col].count()
    w = w.sum()/w
    return w.values.tolist()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CKPT = "./res/base_model=xlm-roberta-base&max_seq_len=250/efficient-galaxy-79/epoch=19-step=93839-val_loss=0.535-val_f1=0.747.ckpt"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_ckpt', type=str, default=CKPT)

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='The initial learning rate for Adam.')
    parser.add_argument('--cache_dir', type=str, default="./.cache", help="huggingface cahce dir")
    parser.add_argument('--dataroot', type=str, default="/mnt/datasets/open")
    parser.add_argument('--num_classes', type=int, default=46)
    # wandb related
    parser.add_argument('--project', type=str, default='dacon-open-lstm')

    return parser

class OpenSeqDataset(Dataset):
    def __init__(self, df, dim=768, labels=None, is_training=True):
        self.df = df
        max_seq_len = self.df['seq_len'].max()
        self.labels = labels
        self.zeros = torch.zeros(max_seq_len, dim)
        self.is_training = is_training

    def __getitem__(self, idx):
        hs, seq_len = self.df.iloc[idx][['hs','seq_len']]
        #if self.is_training:
        #    random.shuffle(hs)
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
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--initializer_range', type=float, default=0.02)
        parser.add_argument('--loss_fct', choices=['cross_entropy', 'f1_loss'], default='cross_entropy')
        return parent_parser

    @staticmethod
    def submission():
        base_ckpt = "./res/base_model=xlm-roberta-base&max_seq_len=250/epoch=10-step=51611-val_loss=0.441-val_f1=0.739.ckpt"
        lstm_ckpt = "./res/lstm_seq/epoch=5-step=13072-val_loss=0.508-val_f1=0.752.ckpt"
        test_cache_path = "prep/test_base.pkl"
        cache_path = "prep/lstm_seq_test.pkl"

        df = with_cache(LitBaseModel.extract_feature, cache_path)(base_ckpt, test_cache_path=test_cache_path)
        ds = OpenSeqDataset(df)
        dl = DataLoader(ds, batch_size=128, shuffle=False)

        m, _ = LitOpenSeq.load_from_ckpt(lstm_ckpt)
        m.eval().cuda().freeze()

        preds = []
        with torch.no_grad():
            for batch in tqdm(dl):
                pred = m(batch)
                preds.append(pred)
        preds = torch.cat(preds)
        df['label'] = preds.cpu().numpy()
        sub_df = df['label']
        sub_df = sub_df.reset_index()
        sub_df.to_csv('sub.csv', index=False)

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

        kwargs = {'num_classes': self.hparams.num_classes, 'average':'macro'}
        self.val_f1 = torchmetrics.F1(**kwargs)
        self.val_acc = torchmetrics.Accuracy(**kwargs)
        self.val_recall = torchmetrics.Recall(**kwargs)
        self.val_precision = torchmetrics.Precision(**kwargs)
        bn = os.path.splitext(os.path.basename(self.hparams.base_ckpt))[0]
        self.cache_path = os.path.join("./feats", f"feat_{bn}.pkl")

        self.lstm = nn.LSTM(
            input_size=self.hparams.input_size, # dim
            hidden_size=self.hparams.hidden_size,
            batch_first=True,
            num_layers=self.hparams.num_layers,
            bidirectional=True)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.hparams.dropout_prob)
        self.lin = nn.Linear(self.hparams.hidden_size, 512)
        self.clf = nn.Linear(512, self.hparams.num_classes)

        self._init_weights(self.lin)
        self._init_weights(self.clf)

        if self.hparams.loss_fct == 'f1_loss':
            self.loss_fct = F1_Loss(self.hparams.num_classes).cuda()
        else:
            self.loss_fct = CrossEntropyLoss()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
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
        with_cache(self.cache_path, LitBaseModel.extract_feature)(self.hparams.base_ckpt)

    def load_datasets(self):
        df = with_cache(self.cache_path, LitBaseModel.extract_feature)(self.hparams.base_ckpt)

        if self.hparams.num_classes == 2:
            df.loc[df.label > 0, 'label'] = 1
        elif self.hparams.num_classes == 45:
            df = df[df.label > 0]
            df.label -= 1

        tr_df, va_df = cv_split(df, self.hparams.cv)

        if self.hparams.loss_fct == 'cross_entropy':
            class_weights = get_class_weights(tr_df)
            assert len(class_weights) == self.hparams.num_classes

            dev = next(self.parameters()).device
            self.loss_fct = CrossEntropyLoss(weight=torch.tensor(class_weights).to(dev))

        tr_ds = OpenSeqDataset(tr_df, labels=tr_df['label'].values.tolist())
        va_ds = OpenSeqDataset(va_df, labels=va_df['label'].values.tolist(), is_training=False)

        return {'train': tr_ds, 'val': va_ds}

    def train_dataloader(self) -> DataLoader:
        ds = self.get_datasets()
        return DataLoader(
            ds['train'],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.get_datasets()['val'], batch_size=self.hparams.batch_size, num_workers=16)

    def get_datasets(self):
        if self._datasets is None:
            self._datasets = self.load_datasets()

        return self._datasets

    def forward(self, batch):
        x, seq_len, labels = batch
        x = x.cuda()
        seq_len = seq_len.cuda()
        logits = self.__step(x, seq_len)
        return {'logits': logits, 'labels': labels}
        #return logits.argmax(dim=1), labels

    def validation_step(self, batch, batch_idx):
        x, seq_len, labels = batch
        logits = self.__step(x, seq_len)
        loss = self.loss_fct(logits, labels)
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
        loss = self.loss_fct(logits, labels)
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
            "lr_scheduler": ReduceLROnPlateau(optimizer, mode='max'),
            "monitor": "val_f1"
        }

    def configure_callbacks(self):
        #early_stop_callback = EarlyStopping(
        #    monitor='val_loss',
        #    min_delta=0.00,
        #    patience=3,
        #    verbose=False,
        #    mode='min'
        #)
        self.print("============ STATE=", self.trainer.state.fn)
        if self.trainer.state.fn == TrainerFn.PREDICTING:
            return []

        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            min_delta=0.00,
            patience=5,
            verbose=False,
            mode='max'
        )
        exp_name = self.trainer.logger.experiment.name
        logger.info(f"{exp_name=}")
        mc = ModelCheckpoint(
            dirpath=f"./res/lstm_seq/{exp_name}",
            monitor="val_acc",
            save_top_k=3,
            filename='{epoch}-{step}-{val_f1:.3f}-{val_loss:.3f}',
            mode='max'
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return [early_stop_callback, mc, lr_monitor]

def predict(ckpt_path):
    assert ckpt_path is not None or name is not None

    ckpt_path = 'a.ckpt'
    model = LitOpenSeq.load_from_checkpoint(ckpt_path)
    trainer = pl.Trainer(gpus=1, amp_level='O2', precision=16)

    p = trainer.predict(model, dataloaders=model.val_dataloader(), return_predictions=True)
    logits, labels = zip(*map(operator.itemgetter('logits', 'labels'), p))
    logits = torch.cat(logits).cpu()
    labels = torch.cat(labels).cpu()
    preds = logits.argmax(dim=-1)
    res = classification_report(labels, preds, output_dict=True)

    df = pd.DataFrame()
    for i in range(model.hparams.num_classes):
        df = df.append(res[str(i)], ignore_index=True)

    return {
        'report': df,
        'logits': logits,
        'labels': labels
    }

def dummy():
    ckpt_path = 'a.ckpt'
    model = LitOpenSeq.load_from_checkpoint(ckpt_path)
    trainer = pl.Trainer(gpus=1, amp_level='O2', precision=16)
    p = trainer.predict(model, dataloaders=model.val_dataloader(), return_predictions=True)

    logits, labels = zip(*map(operator.itemgetter('logits', 'labels'), p))
    logits = torch.cat(logits).cpu()
    labels = torch.cat(labels).cpu()
    preds = logits.argmax(dim=-1)
    print(classification_report(labels, preds))

    ixs = torch.nonzero(preds0>0).squeeze().tolist()

    org_dataset = model.get_datasets()['val']
    ds = torch.utils.data.Subset(org_dataset, ixs)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    model2 = LitOpenSeq.load_from_checkpoint('b.ckpt')

    trainer = pl.Trainer(gpus=1, amp_level='O2', precision=16)
    p2 = trainer.predict(model2, dataloaders=dl, return_predictions=True)
    logits, _ = zip(*map(operator.itemgetter('logits', 'labels'), p2))
    logits = torch.cat(logits).cpu()
    preds2 = logits.argmax(dim=-1)

    preds[ixs] = preds2 + 1

    print(classification_report(labels, preds))


def train(args):
    model = LitOpenSeq(args)

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

if __name__ == '__main__' and not isin_ipython():
    parser = get_parser()
    parser = LitOpenSeq.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    train(args)
