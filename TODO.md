- xlm & keywords defnitely helpful [o]
- clean text seems helpful [o]
- avg & max & ff pooling & [CLS] pooling
- more cleaning text?
- shorten max_len_seq (seems overfitting)

- pooling sweep
avg max ff [CLS]
             o
        o
 o   
 o   o  o
     o  o 

#### (A) check blanced batch/trimming
- [o] check MolBert (%collate_fn)
- [o] check distribtedwrapper 
- [o] check https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
      try BalanceClassSampler (should be not much different from weightedsampling)
      try DynamicBalanceClas
  seems Wrapper w/ WeightedRandom Sampler have problem
- [ ] check classification report
-     recall/precision imbalance problem!
- imablance dataset prpaers
  - curriculum learning
https://web.kamihq.com/web/viewer.html?state=%7B%22ids%22%3A%5B%221bakX6mLeutpEDVDHi_M2zuqZ7OYG3yJl%22%5D%2C%22action%22%3A%22open%22%2C%22userId%22%3A%22108577906332508460772%22%2C%22resourceKeys%22%3A%7B%7D%7D&kami_user_id=5164017
- dynamic curriculum learning
https://web.kamihq.com/web/viewer.html?state=%7B%22ids%22%3A%5B%2218Uer9efFKPzvr5Sr5DDdWDY1q6El37F9%22%5D%2C%22action%22%3A%22open%22%2C%22userId%22%3A%22108577906332508460772%22%2C%22resourceKeys%22%3A%7B%7D%7D&kami_user_id=5164017

## PLAN
- unserstand what's going on with precision/recall imbalance and it's affect on f1 score
- just focus on learning sampling, (prob calibration if time permitted)
  - https://web.kamihq.com/web/viewer.html?state=%7B%22ids%22%3A%5B%221Lz2HBfsQR4d-eJOB4pgTie8wpFPPfqFO%22%5D%2C%22action%22%3A%22open%22%2C%22userId%22%3A%22108577906332508460772%22%7D&filename=undefined&kami_user_id=5164017
  - https://arxiv.org/pdf/1901.06783.pdf
- seems DistributedSamplerWrapper works fine why unbalnced recall/precision?
- balance batching [o]
- comparision MLM / normal
- additional cleaning text
- LSTM classification
- training with whole data

##### hiearchical?

# problem (likely related to (A)
- low precision & high recall
--------------------------------------------------------
# things don't work
- inlcuding keywords in base model? [o]
- early stopping based on loss / no stopping (w/ correct sampler cross entropy loss aligned with f1)
- check weighted sampler working [o] - obviously better
- soft f1 loss doesn't seem to work well [x]
  perhaps better when blanced
- MLM doesn't really work?

# others
- think about `sampler_test.py`
