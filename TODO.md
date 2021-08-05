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
- [ ] check MolBert (%collate_fn)
- [ ] check https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
- [o] check distribtedwrapper 
  seems Wrapper w/ WeightedRandom Sampler have problem

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
