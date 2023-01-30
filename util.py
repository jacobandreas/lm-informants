import numpy as np

def ngrams(seq, order):
    for i in range(len(seq) - order + 1):
        yield seq[i:i+order]

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
