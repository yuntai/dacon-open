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

    for i in range(len(tokens)):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and not inference_mode:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                token = '[MASK]'
            # 10% randomly change token to random token
            elif prob < 0.9:
                token = random.choice(list(tokenizer.token_to_idx.items()))[0]
                while (token in tokenizer.symbols) or (token == tokens[i]):
                    token = random.choice(list(tokenizer.token_to_idx.items()))[0]
            # -> rest 10% randomly keep current token
            else:
                token = tokens[i]

            # set the replace token and append token to output (we will predict these later)
            try:
                output_label.append(tokenizer.token_to_idx[tokens[i]])
                tokens[i] = token
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.token_to_idx['[UNK]'])
                logger.warning('Cannot find token "{}" in token_to_idx. Using [UNK] instead'.format(tokens[i]))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label
