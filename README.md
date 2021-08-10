- unbalnaced label
- hiearchical label (?)
- F1 macro optimization (nothing much as post processing)
- contrasive using detaled label? (?)
- longer text (transformer model)
- https://arxiv.org/abs/1910.10781
- initialization - difference in LB score (ok) used truncated normal
- https://medium.com/@armandj.olivares/using-bert-for-classifying-documents-with-long-texts-5c3e7b04573d
- filter label 0 first? (two-level hiearchy)
- check Seq model initialization
- loading model, F1 score per class
# how to utilize label information?

# hiearchical
https://towardsdatascience.com/hierarchical-classification-by-local-classifiers-your-must-know-tweaks-tricks-f7297702f8fc

# feature aggergation
- LSTM model 
- transformer

# sample balance problem
- https://stats.stackexchange.com/questions/244630/difference-between-sample-weight-and-class-weight-randomforest-classifier

# F1 score
- https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d

# classification 1/0
- https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py

#TODO: multi seq strategy
### LSTM - does not seem to work well perhaps not enough semantic captured from BERT?
   - f1 around 0.818 (binary case)
   - not running new info it seems

- Trasnformer?
- AVG
- SUM
- Perceptron
- not using [CLS] (o)

# try distilBERT?

# pad_packed_demo
https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec (o)

# TODO: 
LSTM: hyperpareameter search?
- check special token option(add_special_tokens) (o)

- https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143

# NLP augmentation
https://neptune.ai/blog/data-augmentation-nlp

# Longformer
https://colab.research.google.com/github/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb

# unsupervised?
- masked LM
- permute keywords?

# multilabel F1
https://medium.com/synthesio-engineering/precision-accuracy-and-f1-score-for-multi-label-classification-34ac6bdfb404#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjNkZjBhODMxZTA5M2ZhZTFlMjRkNzdkNDc4MzQ0MDVmOTVkMTdiNTQiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2Mjc0NTg5OTMsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjEwODU3NzkwNjMzMjUwODQ2MDc3MiIsImVtYWlsIjoieXVudGFpLmt5b25nQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJuYW1lIjoiWXVudGFpIEt5b25nIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FBVFhBSngtYUxEcTZzaWk3UzNhWXJfVHFmSFhFVEQ0QzNrdzNwdVVlZE53MFE9czk2LWMiLCJnaXZlbl9uYW1lIjoiWXVudGFpIiwiZmFtaWx5X25hbWUiOiJLeW9uZyIsImlhdCI6MTYyNzQ1OTI5MywiZXhwIjoxNjI3NDYyODkzLCJqdGkiOiI2OGI3ZWJiOWQ5YzdkODBkZDU4NTZkMzIzNWEyY2NjZmJjNjYwMTQzIn0.mWodc-36AZQNB_vi6X2VDib0DNtm9Umk_USv-bP0xPZyTmfcV0beGD6G6VhE5ZQ9xnU5YgK5LkKC1YYZSSmgDpFWtOf0UUVxjk-LFT4JubBNW6qrNR6EMx4w4IfTVDQmnxG1QlUHscDc8jwt2zzzULLBDrbbL-lNdGJeLloIUhxhNpPMc_TNHd8fnIfxyGKWcX5fS-NlPqOe_penyPa4O6oM-mpenWLxz1uS4m6WyO2F7xrbm3qByLmKXOh-6qWMMOonA9dPRXRo0Jt1h4yBhTPA-1OMbah_NK0-7EZxMLFbsIAZJa-SDPuOwUl8556EFT0ybiFQ4CnnxHiiTOyAqQ

# loss/acc/f1/unbalanced dataset
Cross-entropy & F1 relationship
- https://stackoverflow.com/questions/59688024/cross-entropy-loss-influence-over-f-score
- https://blog.libove.org/2018/06/13/f1-score-rises-while-loss-keeps-increasing/
- https://stackoverflow.com/questions/53354176/how-to-use-f-score-as-error-function-to-train-neural-networks
- (custom F1 loss) https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77289

# model sizes
[('bert-base-multilingual-cased', 177853440, 177.85344),
 ('xlm-roberta-base', 278043648, 278.043648),
 ('xlm-roberta-large', 559890432, 559.890432), -- too big? no pregressing
 ('monologg/kobert', 92186880, 92.18688), - week
 ('monologg/distilkobert', 27803904, 27.803904)] -week

# masked LM eample
https://stackoverflow.com/questions/63030692/how-do-i-use-bertformaskedlm-or-bertmodel-to-calculate-perplexity-of-a-sentence

# NLP augmentation
https://neptune.ai/blog/data-augmentation-nlp

# MLM with bert doens't seem to work increase score

# LESSONS
- loss weight is very efficient and it seems work better than sampling techniques for this problem
