./results/checkpoint-8442 - best loss but F1 kept decreasing
early stopping based on F1

# with correct get_split & samping
./results/checkpoint-51240
0.62

# LSTM w/ [CLS]
f1 0.699

# Bert BASE no [CLS] max_length(100) CLS Task
- weight decay seems helpful
max_length | f1
200        | 0.729/0.581
250        | 0.734/0.54 (weight decay) / more generality? bert_base_250/epoch=12-val_f1=0.73-val_loss=0.54.ckpt
300        |
500        | 0.649

# MASKED LM
