import random
import logging
import argparse
import re
import argparse
from pathlib import Path
import os
import torch.distributed as dist

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

class LitSimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 2)
        self._datasets = None

    def forward(self, batch):
        return self.lin(batch[0].type(torch.FloatTensor).cuda())

    def validation_step(self, batch, batch_idx):
        inp, targets = batch
        inp = inp.unsqueeze(-1).type(torch.FloatTensor).cuda()
        targets = targets.type(torch.LongTensor).cuda()

        logits = self.lin(inp)
        loss = F.cross_entropy(logits, targets)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        inp, targets = batch
        inp = inp.unsqueeze(-1).type(torch.FloatTensor).cuda()
        targets = targets.type(torch.LongTensor).cuda()

        logits = self.lin(inp)
        loss = F.cross_entropy(logits, targets)
        return loss

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets()['train'], batch_size=8, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets()['val'], batch_size=8, shuffle=False, num_workers=4)

    def datasets(self):
        if self._datasets:
            return self._datasets

        self._datasets = {'train': self.ds[:-200], 'val': self.ds[-200:]}

        return self._datasets

    def setup(self, stage):
        print(f"SETUP({stage=})")
        sz = 2000
        targets = torch.arange(sz)
        labels = torch.multinomial(torch.FloatTensor([0.5, 0.5]), sz, True)

        self.ds = torch.utils.data.TensorDataset(targets, labels)

    def prepare_data(self):
        print("PREPARE_DATA")

    def configure_optimizers(self):
        print("CONFIGURE_OPTIMIZERS")
        optimizer = AdamW(self.lin.parameters(), lr=0.005)
        return optimizer

    def configure_callbacks(self):
        print("CONFIGURE_CALLBACKS")
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            mode="max",
            patience=3,
        )
        v_num = self.trainer.logger.experiment.name
        checkpoint = ModelCheckpoint(
            dirpath=f"./tmp/{v_num}",
            monitor="val_loss",
            save_top_k=3,
            mode='max',
            every_n_epochs=1,
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')

        return [early_stop, checkpoint, lr_monitor]

def fit():
    model = LitSimpleModel()
    wandb_logger = WandbLogger(
        project="simple",
    )

    name = wandb_logger.experiment.name
    if type(name) == str:
        print("exp name=", name)

    trainer = pl.Trainer(
        gpus=1,
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        max_epochs=2,
        checkpoint_callback=False,
        logger=wandb_logger,
        replace_sampler_ddp=False,
        val_check_interval=0.5,
    )
    trainer.fit(model)

if __name__ == '__main__' and not common.isin_ipython():
    fit()

