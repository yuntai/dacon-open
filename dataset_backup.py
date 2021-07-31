import torch
import random
import numpy as np
import os
import torch.nn.functional as F
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# from github.com:BenevolentAI/MolBERT.git
def random_word(tokens, tokenizer, inference_mode: bool = False):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.

    Args:
        tokens: list of str, tokenized sentence.
        tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        inference_mode: if True, don't do any input modifications. Used at inference time.

    Returns
        tokens: masked tokens
        output_label: labels for LM prediction
    """
    output_label = []

    mask_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index(tokenizer.special_tokens_map['mask_token'])]

    for i in range(len(tokens)):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and not inference_mode:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                #token = '[MASK]'
                token = mask_id
            # 10% randomly change token to random token
            elif prob < 0.9:
                token = random.choice(list(tokenizer.get_vocab().items()))[1]
                #token = random.choice(list(tokenizer.token_to_idx.items()))[0]
                #while (token in tokenizer.symbols) or (token == tokens[i]):
                while (token in tokenizer.all_special_ids) or (token == tokens[i]):
                    token = random.choice(list(tokenizer.get_vocab().items()))[1]
                    #token = random.choice(list(tokenizer.token_to_idx.items()))[0]
            # -> rest 10% randomly keep current token
            else:
                token = tokens[i]

            # set the replace token and append token to output (we will predict these later)
            try:
                #output_label.append(tokenizer.token_to_idx[tokens[i]])
                output_label.append(token)
                tokens[i] = token
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.token_to_idx['[UNK]'])
                logger.warning('Cannot find token "{}" in token_to_idx. Using [UNK] instead'.format(tokens[i]))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

class OpenMLMDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_seq_len=512, add_speical_tok=False, seed=42):
        super().__init__()
        self.df = df.copy()
        self.max_seq_len = max_seq_len
        self.add_speical_tok = add_speical_tok
        self.tokenizer = tokenizer
        self.seed = seed

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        np.random.shuffle(row.ixs)
        toks = row.toks[row.ixs]
        tok_lens = row.tok_lens[row.ixs]
        toks = np.concatenate(toks).astype(int)
        toks = torch.tensor(toks)
        if toks.size(0) > self.max_seq_len:
            ovf = self.max_seq_len - toks.size(0)
            if np.random.randint(2) == 1:
                toks = toks[:ovf]
            else:
                toks = toks[-ovf:]

        toks, labels = random_word(toks, self.tokenizer)
        labels = torch.tensor(labels)
        sz = toks.size(0)
        attention_mask = torch.zeros(self.max_seq_len)
        attention_mask[:sz] = 1

        toks = F.pad(toks, (0, self.max_seq_len-sz))
        labels = F.pad(labels, (0, self.max_seq_len-sz))

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
