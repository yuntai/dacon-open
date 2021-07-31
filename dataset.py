import torch
import random
import numpy as np
import os
import torch.nn.functional as F
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class OpenMLMDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_seq_len=512, add_speical_tok=False, seed=42):
        super().__init__()
        self.df = df.copy()
        self.max_seq_len = max_seq_len
        self.add_speical_tok = add_speical_tok
        self.tokenizer = tokenizer
        self.seed = seed
        self.mask_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index(tokenizer.special_tokens_map['mask_token'])]

    def random_word(self, tokens):
        probs = np.random.rand(len(tokens))
        labels = tokens.copy()
        ix = probs < 0.15
        tokens[ix] = self.mask_id
        labels[~ix] = -1
        return tokens, labels

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        np.random.shuffle(row.ixs)
        toks = row.toks[row.ixs]
        tok_lens = row.tok_lens[row.ixs]
        toks = np.concatenate(toks).astype(int)

        if toks.size > self.max_seq_len:
            ovf = self.max_seq_len - toks.size
            if np.random.randint(2) == 1:
                toks = toks[:ovf]
            else:
                toks = toks[-ovf:]

        toks, labels = self.random_word(toks)
        toks = torch.tensor(toks)
        labels = torch.tensor(labels)

        sz = toks.size(0)
        attention_mask = torch.zeros(self.max_seq_len)
        attention_mask[:sz] = 1

        pad_sz = self.max_seq_len-sz
        toks = F.pad(toks, (0, pad_sz))
        labels = F.pad(labels, (0, pad_sz))

        return {'input_ids': toks, 'attention_mask': attention_mask, 'labels': labels}

    def __len__(self):
        return self.df.shape[0]

def prep_txt(df, tokenizer):
    from tqdm.auto import tqdm

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

    cols = ['index', 'toks', 'tok_lens', 'tot_len', 'ixs']
    if 'label' in df.columns:
        cols += ['label']

    df = df[cols]

    return df
