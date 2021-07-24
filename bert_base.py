# some code snippets from and restructued ala 
# https://github.com/BenevolentAI/MolBERT/blob/main/molbert/models/base.py

import random
import logging
import argparse
import re
import argparse
from pathlib import Path

import torch
import torch.nn as nn

import pandas as pd
import numpy as np

import torchmetrics
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from pytorch_lightning.loggers import WandbLogger

from transformers import BertTokenizerFast, BertModel, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from transformers import AdamW
from tqdm.auto import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import torch.distributed as dist

import common

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BertBaseClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        config = self.bert.config

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)

        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()


    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        x = output['last_hidden_state'].mean(axis=1)
        x = self.activation(x)

        x = self.dropout(x)
        logits = self.classifier(x)

        return logits

class LitBertBase(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters(hparams)

        num_labels = 46
        self.model = BertBaseClassifier(args, num_labels)
        self.valid_f1 = torchmetrics.F1(num_classes=num_labels, average='macro')
        self.valid_acc = torchmetrics.Accuracy()

        self._datasets = None

    def forward(self, x):
        return self.model(x).argmax(dim=1)

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits = self.model(**batch)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        preds = logits.argmax(dim=1)
        self.valid_acc(preds, labels)
        self.valid_f1(preds, labels)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.valid_f1, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        labels = batch.pop("labels")
        logits = self.model(**batch)
        loss = F.cross_entropy(logits, labels)
        return loss

    def train_dataloader(self) -> DataLoader:
        from sampler import DistributedSampler

        w = self.get_datasets()['weights']
        weighted_sampler = WeightedRandomSampler(w, len(w))

        return DataLoader(
            self.get_datasets()['train'],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=DistributedSampler(weighted_sampler),
            pin_memory=True,
            num_workers=16)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.get_datasets()['val'], batch_size=self.hparams.batch_size, num_workers=16)

    # @property raises torch.nn.modules.module.ModuleAttributeError
    def get_datasets(self):
        if self._datasets is None:
            self._datasets = self.load_datasets()

        return self._datasets

    def prepare_data(self):
        from common import prep
        prep(seed=self.hparams.seed, max_len=self.hparams.max_len)

    def load_datasets(self):
        from common import prep, cv_split, get_weights, get_dataset

        df = prep(seed=self.hparams.seed, max_len=self.hparams.max_len)
        num_classes = df.label.nunique()
        assert sorted(df.label.unique().tolist()) == list(range(num_classes))

        tr_df, va_df = cv_split(df, self.hparams.cv)

        tr_df = get_weights(tr_df)
        weights = tr_df.pop('w').values.tolist()
        self.sample_weights = weights

        tr_ds = get_dataset(tr_df)
        va_ds = get_dataset(va_df)

        return {'train': tr_ds, 'val': va_ds, 'weights': weights}

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

        num_batches = len(self.get_datasets()['train']) // self.hparams.batch_size
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

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            mode="min",
            patience=3,
        )
        checkpoint = ModelCheckpoint(
            dirpath=f"./res/bertbase_max_len={self.hparams.max_len}",
            monitor="val_f1",
            save_top_k=5,
            filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}',
            mode='max'

        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return [early_stop, checkpoint, lr_monitor]

def train_bert_base(args):
    model = LitBertBase(args)

    wandb_logger = WandbLogger(name=args.wandb_name)
    trainer = pl.Trainer(
        gpus=args.gpus,
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        max_epochs=args.max_epochs,
        checkpoint_callback=False,
        logger=wandb_logger,
    )
    trainer.fit(model)

if __name__ == '__main__':
    from common import get_parser
    args = get_parser()
    args.num_labels = 46
    pl.seed_everything(args.seed)
    train_bert_base(args)
