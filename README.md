- unbalnaced label
- hiearchical label
- F1 macro optimization
- contrasive using detaled label? NA
- longer text 
- https://arxiv.org/abs/1910.10781
- initialization - difference in LB score 
  used truncated normal
- https://medium.com/@armandj.olivares/using-bert-for-classifying-documents-with-long-texts-5c3e7b04573d
- filter label 0 first? (two-level hiearchy)
- check Seq model initialization
- loading model, F1 score per class

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
   - f1 around 0.818 (46 labels)
   - not running new info it seems

- Trasnformer?
- AVG
- SUM
- Perceptron
- not using [CLS]

# relationship btw binary & cross_entropy
# try distilBERT?

# pad_packed_demo
https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec

# TODO: 
LSTM: hyperpareameter search?
check special token option(add_special_tokens)

- https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143

# NLP augmentation
https://neptune.ai/blog/data-augmentation-nlp

# Longformer
https://colab.research.google.com/github/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb

# unsupervised?
- masked LM
