- xlm & keywords defnitely helpful
- clean text seems helpful
- avg & max & ff pooling & [CLS] pooling

- pooling sweep
avg max ff [CLS]
             o
        o
 o   
 o   o  o
     o  o 

- check blanced batch/trimming
check MolBert (%collate_fn)
check distribtedwrapper

- hiearchical?

# problem
- low precision & high recall
- check final classification report
--------------------------------------------------------
# things don't work
- inlcuding keywords in base model? [o]
- early stopping based on loss / no stopping (w/ correct sampler cross entropy loss aligned with f1)
- check weighted sampler working [o] - obviously better
- soft f1 loss doesn't seem to work well [x]


