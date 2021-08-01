import torch
import argparse
import random
import numpy as np
import os
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BertForMaskedLM, AutoTokenizer
from typing import Optional
import pandas as pd
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='The initial learning rate for Adam.')
    return parser

class LitMLModel(pl.LightningModule):
    @staticmethod
    def extract_feature(ckpt_path):
        from common import prep, with_cache, get_dataset
        from tqdm.auto import tqdm
        import pathlib
        model = LitMLModel.load_from_checkpoint(ckpt_path)
        model.eval().cuda().freeze()
        args = model.hparams
        args.base_model = 'bert-base-multilingual-cased'
        args.cache_dir = './.cache'
        args.use_keywords = True
        args.use_english = True
        args.cv_size = 5
        args.cv = 0
        args.seed = 42
        args.dataroot = pathlib.Path("/mnt/datasets/open")
        args.max_seq_len = 250

        df = with_cache(prep, './prep/mlm_feat.pkl')(args)

        ds = get_dataset(df)
        dl = DataLoader(ds, batch_size=32, shuffle=False)

        hidden_states = []
        with torch.no_grad():
            for batch in tqdm(dl):
                batch.pop('labels')
                item = {k:v.to('cuda') for k, v in batch.items()}
                item['output_hidden_states'] = True
                out = model(item)
                hidden_states.append(out['hidden_states'][-1].mean(axis=1).cpu())

        hs = torch.cat(hidden_states)
        hs = list(map(lambda x: torch.squeeze(x).numpy(), torch.split(hs, 1)))
        df['hs'] = hs

        hs_df = df.groupby('index')['hs'].apply(list)
        df.pop('hs')

        cols = ['index', 'label', 'cv']
        df = df.groupby('index')[cols].first().set_index('index').merge(hs_df, left_index=True, right_index=True, how='inner')
        df['seq_len'] = df.hs.apply(len)
        df['hs'] = df['hs'].apply(torch.tensor)

        return df

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased', cache_dir='./.cache')

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = out['loss']
        return loss

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

        ___sz = 174304
        num_batches = ___sz // self.hparams.batch_size
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


class OpenDataModule(pl.LightningDataModule):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        print("prepare_data")
        from common import with_cache
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        with_cache(prep_txt, 'mlm_prep.pkl')(tokenizer)

    def setup(self, stage: Optional[str] = None):
        print("stage=", stage)
        from common import with_cache
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.df = with_cache(prep_txt, 'mlm_prep.pkl')(tokenizer)
        self.df = self.df[self.df.cv != 0]
        self.ds = OpenMLMDataset(self.df, tokenizer)

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size)
    #def val_dataloader(self):
    #    return DataLoader(self.ds, batch_size=self.hparams.batch_size)

    #def test_dataloader(self):
    #    return DataLoader(self.ds, batch_size=self.hparams.batch_size)


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
        labels[~ix] = -100
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
        labels = F.pad(labels, (0, pad_sz), value=-100)

        return {'input_ids': toks, 'attention_mask': attention_mask, 'labels': labels}

    def __len__(self):
        return self.df.shape[0]


def prep_txt(tokenizer):
    from tqdm.auto import tqdm
    from common import prep_cv

    df = pd.read_csv("/mnt/datasets/open/train.csv")
    df = prep_cv(df)

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

    cols = ['index', 'toks', 'tok_lens', 'tot_len', 'ixs', 'cv']
    if 'label' in df.columns:
        cols += ['label']

    df = df[cols]

    return df

def fit():
    parser = get_parser()
    args = parser.parse_args()
    dm = OpenDataModule(args.batch_size)
    model = LitMLModel(args)
    trainer = pl.Trainer(
        gpus=args.gpus,
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        max_epochs=args.max_epochs
    )
    trainer.fit(model, dm)
