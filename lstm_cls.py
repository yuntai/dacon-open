import random
import argparse
import re
from pathlib import Path

import torch
import torch.nn as nn

import pandas as pd
import numpy as np

#from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torchmetrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
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

from sklearn.metrics import classification_report
from  torch.nn.utils.rnn import pack_padded_sequence

MODEL_NAME = 'bert-base-multilingual-cased'
CACHE_DIR = 'bert_ckpts'

MAX_LEN = 200
TOK_COLS = ['input_ids', 'token_type_ids', 'attention_mask']

#parser = argparse.ArgumentParser()
#parser.add_argument('--dataroot', '-d', type=str, default='/mnt/

def cache_df(fn):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if Path(fn).exists():
                print(f"reading from {fn} ...")
                df = pd.read_pickle(fn)
            else:
                df = func(*args, **kwargs)
                print(f"saving to {fn} ...")
                df.to_pickle(fn)
            return df
        return wrapper
    return decorator

def get_split(text1, chunk_sz=200, overlap_sz=50):
    words = text1.split()

    step_sz = chunk_sz - overlap_sz

    res = [words[:chunk_sz]]
    if len(words) > chunk_sz:
        res += [words[i:i+chunk_sz] for i in range(step_sz, len(words), step_sz)]

    return list(map(lambda x: " ".join(x), res))

# create torch dataset
class OpenDataset(Dataset):
    def __init__(self, df, labels=None):
        self.df = df
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.df.iloc[idx].to_dict().items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.df.shape[0]

def clean_text(s):
    #s = re.sub("[^가-힣ㄱ-하-ㅣ]", " ", s)
    s = re.sub('[^A-Za-z가-힣ㄱ-하-ㅣ]+', ' ', s)
    s = re.sub("(\\W)+"," ", s)
    return s.strip()

def prep_txt(df):

    cols = ['과제명', '요약문_연구목표', '요약문_연구내용', '요약문_기대효과', '요약문_한글키워드', '요약문_영문키워드']

    df = df[cols + ['index','label']].copy()
    df.fillna(' ', inplace=True)

    df['data'] = df[cols[0]]
    for c in cols[1:]:
        df['data'] += ' ' + df[c]

    df['data_cleaned'] = df['data'].apply(clean_text)

    return df

# split train data to 5 set
def prep_cv(df, cv=5):
    indices = list(df.index)
    random.shuffle(indices)
    sz = df.shape[0]//cv
    for ix in range(cv-1):
        df.loc[indices[ix*sz:(ix+1)*sz], 'cv'] = ix
    df.loc[indices[(ix+1)*sz:], 'cv'] = ix+1
    df['cv'] = df.cv.astype(int)

    return df

def prep_explode(df):
    df['data_split'] = df['data_cleaned'].apply(get_split)

    train_l, label_l, cv_l, index_l, subix_l = [], [], [], [], []

    for _, row in df.iterrows():
      for ix, l in enumerate(row['data_split']):
          train_l.append(l)
          label_l.append(row['label'])
          cv_l.append(row['cv'])
          index_l.append(row['index'])
          subix_l.append(ix)

    return pd.DataFrame({'index': index_l, 'subix': subix_l, 'data':train_l, 'label':label_l, 'cv': cv_l})

# tokenize
def prep_tok(df):
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, do_lower_case=False)
    kwargs = {
        'add_special_tokens': True,
        'max_length': MAX_LEN,
        'pad_to_max_length': True,
        'return_attention_mask': True,
        'truncation': True
    }
    toks = tokenizer(df['data'].values.tolist(), **kwargs)
    for k in TOK_COLS:
        df[k] = toks[k]

    return df

# load dataset, clean text, assing cv and tokenize
def prep(seed):
    dataroot = Path('/mnt/datasets/open')
    train_df = pd.read_csv(dataroot/'train.csv')

    df = prep_txt(train_df)
    df = prep_cv(df)
    df = prep_explode(df)
    df = prep_tok(df)

    return df

def get_dataset(df):
    X = df[TOK_COLS]
    y = df['label'].values.tolist()
    return OpenDataset(X, y)

# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    #accuracy = accuracy_score(y_true=labels, y_pred=pred)
    #recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    #precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {"f1": f1}
    #return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# TODO: order preserving way to assing weights
def get_weights(df, col='label'):
    w = df.groupby(col)[col].count()
    w = w.sum()/w
    w.name = 'w'
    # ORDER CHANGES!
    return df.merge(w, how='left', left_on='label', right_index=True)

def get_trainer_klass(weights):
    sz = len(weights)
    class WeightSamplingTrainer(Trainer):
        def get_train_dataloader(self) -> DataLoader:
            train_sampler = WeightedRandomSampler(weights, sz)
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    return WeightSamplingTrainer

def cv_split(df, cv=0):
    tr_df = df.loc[df.cv!=cv]
    va_df = df.loc[df.cv==cv]

    return tr_df, va_df

class OpenClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

        config = self.bert.config
        self.num_labels = num_labels

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.loss_fn = nn

        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()


    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        x = output['last_hidden_state'].mean(axis=1)
        x = self.activation(x)

        x = self.dropout(x)
        logits = self.classifier(x)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss.unsqueeze(0)
            return (loss, logits,)
        else:
            return logits

class LitBertSem(pl.LightningModule):
    def __init__(self, num_labels):
        super().__init__()
        self.model = OpenClassifier(num_labels)
        self.num_labels = num_labels

    def forward(self, x):
        return self.model(x).argmax(dim=1)

    def validation_step(self, batch, batch_idx):
        x, seq_len, labels = batch
        logits = self.classifier(x, seq_len)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        preds = logits.argmax(dim=1)
        acc = torchmetrics.functional.accuracy(preds, labels)
        f1 = torchmetrics.functional.f1(preds, labels, average='macro', num_classes=self.num_labels)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, labels = batch
        logits = self.classifier(x, seq_len)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        num_epochs = 10
        num_training_steps = num_epochs * len(tr_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_f1"
        }


def train_custom_cv(cv=0, seed=42, result_dir="./results"):
    df = prep()
    num_labels = df.label.nunique()

    tr_df, va_df = cv_split(df, cv)

    tr_df = get_weights(tr_df)
    weights = tr_df.pop('w').values.tolist()

    tr_ds = get_dataset(tr_df)
    va_ds = get_dataset(va_df)

    train_sampler = WeightedRandomSampler(weights, len(weights))
    tr_loader = DataLoader(
        tr_ds,
        batch_size=36,
		shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=16)
    va_loader = DataLoader(va_ds, batch_size=18)


    model = OpenClassifier(num_labels)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 10
    num_training_steps = num_epochs * len(tr_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    device = torch.device("cuda")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in tr_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

def train_cv(cv=0, seed=42):
    df = prep()
    num_labels = df.label.nunique()

    tr_df, va_df = cv_split(df, cv)

    tr_df = get_weights(tr_df)
    weights = tr_df.pop('w').values.tolist()

    tr_ds = get_dataset(tr_df)
    va_ds = get_dataset(va_df)

    tr_loader = DataLoader(tr_ds, shuffle=True, batch_size=36)
    va_loader = DataLoader(va_ds, batch_size=18)

    #tr_ds, va_ds = get_datasets(tr_df, va_df)

    m = OpenClassifier(num_labels)
    #cls_model = BertModel.from_pretrained(
    #    MODEL_NAME,
    #    cache_dir=CACHE_DIR,
    #    num_labels=num_labels)

    # Define Trainer
    args = TrainingArguments(
        output_dir="./results_open",
        evaluation_strategy="epoch",
        #eval_steps=500,
        per_device_train_batch_size=36,
        per_device_eval_batch_size=18,
        num_train_epochs=10,
        seed=seed,
        load_best_model_at_end=True
    )

    trainer_klass = get_trainer_klass(weights)

    trainer = trainer_klass(
        model=m,
        args=args,
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train pre-trained model
    trainer.train()

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

class LSTMClassifier(nn.Module):
    def __init__(self, dim=768, hidden_sz=1024, num_labels=46, dropout_prob=0.1, initializer_range=0.02, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_sz,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=True)

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_prob)
        self.lin = nn.Linear(hidden_sz, 512)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(512, num_labels)

        self.classifier.weight.data.normal_(mean=0.0, std=initializer_range)
        self.classifier.bias.data.zero_()

    def forward(self, x, seq_len):
        packed_input = pack_padded_sequence(x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.lstm(packed_input)

        x = self.activation(ht[-1])
        x = self.dropout(x)
        x = self.lin(x)
        logits = self.classifier(x)

        return logits

class LitOpenSequence(pl.LightningModule):

    def __init__(self, lr, num_labels=46):
        super().__init__()
        self.classifier = LSTMClassifier(num_labels=num_labels)
        self.lr = lr
        self.num_labels = num_labels
        self.valid_f1 = torchmetrics.F1(num_classes=num_labels, average='macro')
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.classifier(x).argmax(dim=1)

    def validation_step(self, batch, batch_idx):
        x, seq_len, labels = batch
        logits = self.classifier(x, seq_len)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        preds = logits.argmax(dim=1)
        self.valid_acc(preds, labels)
        self.valid_f1(preds, labels)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.valid_f1, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, seq_len, labels = batch
        logits = self.classifier(x, seq_len)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(optimizer),
            "monitor": "val_f1"
        }

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor='val_f1',
            min_delta=0.00,
            patience=5,
            verbose=False,
            mode='max'
        )
        mc = ModelCheckpoint(
            monitor='val_f1',
            save_top_k=5,
            dirpath='./res_seq',
            filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}',
            mode='max'
        )
        return [early_stop_callback, mc]


def feature_extract(ckpt, seed=42, cv=0):
    df = prep(seed)

    ds = get_dataset(df)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=16)

    m = BertForSequenceClassification.from_pretrained(ckpt)
    m.eval()
    m.cuda()
    hidden_states = []
    with torch.no_grad():
        for x in loader:
            item = {k:v.to('cuda') for k, v in x.items()}
            res = m(**item, output_hidden_states=True)
            # last hidden state, first token [CLS]
            s = res.hidden_states[-1][:,0,:]
            hidden_states.append(s.cpu())

    hs = sum([list(map(torch.squeeze, h.split(1))) for h in hidden_states], [])
    hs = [h.numpy() for h in hs]
    df['hs'] = hs
    hs_df = df.groupby('index')['hs'].apply(list)
    _ = df.pop('hs')

    df0 = df.groupby('index')[['index','label','cv']].first().set_index('index').merge(hs_df, left_index=True, right_index=True, how='inner')
    df0['seq_len'] = df0.hs.apply(len)
    df0['hs'] = df0['hs'].apply(torch.tensor)

    return df0

def classifier_train(do_lr_find=False, num_labels=46):
    ckpt = './results/checkpoint-51240'
    cv = 0
    df = feature_extract(ckpt, cv=0)
    if num_labels == 2:
        df.loc[df.label != 0, 'label'] = 1

    tr_df = df[df.cv != cv]
    va_df = df[df.cv == cv]

    tr_df = get_weights(tr_df)
    weights = tr_df.pop('w').values.tolist()

    tr_ds = OpenSeqDataset(tr_df, labels=tr_df['label'].values.tolist())
    va_ds = OpenSeqDataset(va_df, labels=va_df['label'].values.tolist())

    train_sampler = WeightedRandomSampler(weights, len(weights))
    tr_dl = DataLoader(tr_ds, batch_size=32, num_workers=16, sampler=train_sampler, shuffle=False)
    va_dl = DataLoader(va_ds, batch_size=64, num_workers=16)

    lr = 0.00001
    if do_lr_find:
        model = LitOpenSequence(num_labels=num_labels, lr=lr)
        trainer = pl.Trainer(gpus=1)
        trainer.tune(model, train_dataloader=tr_dl)
        lr_finder = trainer.tuner.lr_find(model, train_dataloader=tr_dl)
        lr = lr_finder.suggestion()
        print(f"new {lr=}")

    wandb_logger = WandbLogger(name="lstm_seq")

    model = LitOpenSequence(lr=lr, num_labels=num_labels)
    trainer = pl.Trainer(gpus=1,
                         gradient_clip_val=0.5,
                         logger=wandb_logger)
    trainer.fit(model, tr_dl, va_dl)

    return trainer

def eval_model(ckpt, cv=0, va_loader=None):
    if va_loader is None:
        df = prep()

        _, va_df = cv_split(df, cv)
        va_ds = get_dataset(va_df)
        va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)

    m = BertForSequenceClassification.from_pretrained(ckpt)
    m.eval()
    m.cuda()
    preds, labels = [], []
    hidden_states = []
    with torch.no_grad():
        for x in va_loader:
            label = x.pop('labels')
            item = {k:v.to('cuda') for k, v in x.items()}
            res = m(**item)
            preds.append(res['logits'].cpu().argmax(axis=-1))
            labels.append(label)
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    print(classification_report(labels, preds))
    output = [labels, preds]
    if output_hidden_states:
        output += [hidden_states]
    return tuple(output)

classifier_train(do_lr_find=False)
