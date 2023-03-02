import numpy as np

def ngrams(seq, order):
    for i in range(len(seq) - order + 1):
        yield seq[i:i+order]

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def entropy(probs, length_norm=False):
    ent = -probs * np.log(probs) - (1-probs) * np.log(1-probs)
    if length_norm:
        return ent.sum()/len(ent)
    else:
        return ent.sum()

