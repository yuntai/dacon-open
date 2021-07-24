from pathlib import Path
import pandas as pd
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import re
import torch
import random
import functools

TOK_COLS = ['input_ids', 'token_type_ids', 'attention_mask']

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

def get_split(text1, chunk_size=250, overlap_pct=0.25):
    words = text1.split()
    overlap_sz = int(overlap_pct * chunk_size)

    step_size = chunk_size - overlap_sz

    res = [words[:chunk_size]]
    if len(words) > chunk_size:
        res += [words[i:i+chunk_size] for i in range(step_size, len(words), step_size)]

    return list(map(lambda x: " ".join(x), res))

def cache_df(fntmpl):
    def decorator(func):
        def wrapper(*args, **kwargs):
            fn = fntmpl.format(**kwargs)
            print(f"{fn}" +  (' ' if Path(fn).exists() else ' NOT ') + "FOUND")
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

def clean_text(s):
    #s = re.sub("[^가-힣ㄱ-하-ㅣ]", " ", s)
    s = re.sub('[^A-Za-z가-힣ㄱ-하-ㅣ]+', ' ', s)
    s = re.sub("(\\W)+"," ", s)
    return s.strip()


def get_weights(df, col='label'):
    w = df.groupby(col)[col].count()
    w = w.sum()/w
    w.name = 'w'
    # ORDER CHANGES?
    return df.merge(w, how='left', left_on='label', right_index=True)

def cv_split(df, cv):
    tr_df = df.loc[df.cv!=cv]
    va_df = df.loc[df.cv==cv]

    return tr_df, va_df

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
def prep_cv(df, cv=5, seed=42):
    indices = list(df.index)
    random.Random(seed).shuffle(indices)
    sz = df.shape[0]//cv
    for ix in range(cv-1):
        df.loc[indices[ix*sz:(ix+1)*sz], 'cv'] = ix
    df.loc[indices[(ix+1)*sz:], 'cv'] = ix+1
    df['cv'] = df.cv.astype(int)

    return df

def prep_explode(df, max_len):
    __get_split = functools.partial(get_split, chunk_size=max_len)
    df['data_split'] = df['data_cleaned'].apply(__get_split)

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
def prep_tok(df, add_special_tokens=False):
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, do_lower_case=False)
    kwargs = {
        'add_special_tokens': add_special_tokens,
        'padding': True,
        'return_attention_mask': True,
        'truncation': True
    }
    toks = tokenizer(df['data'].values.tolist(), **kwargs)
    for k in TOK_COLS:
        df[k] = toks[k]

    return df

# load dataset, clean text, assing cv and tokenize
@cache_df('./prep/baseprep_seed={seed}_maxlen={max_len}.pkl')
def prep(seed=None, max_len=None):
    assert seed is not None and max_len is not None
    dataroot = Path('/mnt/datasets/open')
    train_df = pd.read_csv(dataroot/'train.csv')

    df = prep_txt(train_df)
    df = prep_cv(df, seed=seed)
    df = prep_explode(df, max_len)
    df = prep_tok(df)

    return df

def get_dataset(df):
    X = df[TOK_COLS]
    y = df['label'].values.tolist()
    return OpenDataset(X, y)
