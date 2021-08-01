from pathlib import Path
import argparse
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import re
import torch
import random
import functools

#TOK_COLS = ['input_ids', 'token_type_ids', 'attention_mask']
TOK_COLS = ['input_ids', 'attention_mask']

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

def clean_text(s):
    #s = re.sub("[^가-힣ㄱ-하-ㅣ]", " ", s)
    #s = re.sub('[^A-Za-z가-힣ㄱ-하-ㅣ]+', ' ', s)
    s = re.sub("(\\W)+"," ", s)
    return s.strip()

def get_weights(df, col='label'):
    w = df.groupby(col)[col].count()
    w = w.sum()/w
    w.name = 'w'
    return df.merge(w, how='left', left_on='label', right_index=True)

def cv_split(df, cv):
    tr_df = df.loc[df.cv!=cv]
    va_df = df.loc[df.cv==cv]

    return tr_df, va_df

def prep_txt(df, include_keywords=True, include_english=True, is_test=False):

    cols = ['과제명', '요약문_연구목표', '요약문_연구내용', '요약문_기대효과']
    if include_keywords:
        cols += ['요약문_한글키워드']
        if include_english:
            cols += ['요약문_영문키워드']

    if is_test:
        df = df[cols + ['index']].copy()
    else:
        df = df[cols + ['index','label']].copy()
    df.fillna(' ', inplace=True)

    df['data'] = df[cols[0]]
    for c in cols[1:]:
        df['data'] += ' ' + df[c]

    df['data_cleaned'] = df['data'].apply(clean_text)

    return df

# split train data to 5 set
def prep_cv(df, cv_size=5, seed=42):
    indices = list(df.index)
    random.Random(seed).shuffle(indices)
    sz = df.shape[0]//cv_size
    for ix in range(cv_size-1):
        df.loc[indices[ix*sz:(ix+1)*sz], 'cv'] = ix
    df.loc[indices[(ix+1)*sz:], 'cv'] = ix+1
    df['cv'] = df.cv.astype(int)

    return df

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

# tokenize
def prep_tok(df, tokenizer, add_special_tokens=False):
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

def get_tokenizer(base_model, **kwargs):
    if base_model in ['monologg/kobert', 'monologg/distilkobert']:
        from tokenization_kobert import KoBertTokenizer
        tokenizer = KoBertTokenizer.from_pretrained(base_model, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, **kwargs)

    return tokenizer

def with_cache(func, cache_path):
    def __inner(*args, **kwargs):
        if Path(cache_path).exists():
            print(f"FOUND {cache_path}")
            df = pd.read_pickle(cache_path)
        else:
            print(f"NOT FOUND {cache_path}")
            df = func(*args, **kwargs)
            print(f"saving to {cache_path} ...")
            df.to_pickle(cache_path)
        return df
    return __inner

def prep(args, is_test=False):
    fn = 'test.csv' if is_test else 'train.csv'
    df = pd.read_csv(args.dataroot/fn)
    tokenizer = get_tokenizer(args.base_model, cache_dir=args.cache_dir, do_lower_case=False)

    df = prep_txt(df, args.use_keywords, args.use_english, is_test=is_test)
    if not is_test:
        df = prep_cv(df, cv_size=args.cv_size, seed=args.seed)
    df = prep_explode(df, args.max_seq_len, is_test=is_test)
    df = prep_tok(df, tokenizer)
    return df

def get_dataset(df, is_test=False):
    X = df[TOK_COLS]
    y = None
    if not is_test:
        y = df['label'].values.tolist()
    return OpenDataset(X, y)
