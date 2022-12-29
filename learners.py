from collections import Counter
from functools import reduce
import itertools as it
import numpy as np
import scorers
from tqdm import tqdm
from scipy.special import logsumexp

import torch
from torch import nn, optim

class VBLearner():
    def __init__(self, dataset, strategy, phoneme_features, feature_vocab):
        self.dataset = dataset
        self.strategy = strategy 
        self.phoneme_features = phoneme_features
        self.feature_vocab = feature_vocab

    def initialize(self):
        self.hypothesis = scorers.MeanFieldScorer(
            self.dataset,
            self.phoneme_features,
            self.feature_vocab,
        )
        self.observations = []

    def observe(self, seq, judgment, update=True):
        assert update
        self.observations.append((seq, judgment))
        self.hypothesis.update(seq, judgment)

    def cost(self, seq):
        return self.hypothesis.cost(seq)

    def entropy(self, seq):
        return self.hypothesis.entropy(seq)

    def full_nll(self, data):
        if len(data) == 0:
            #return np.nan
            return 0
        else:
            return sum(self.hypothesis.logprob(s, j) for s, j in data)

    def discriminate(self, good_data, bad_data):
        scored = []

        good_data = [(d, True) for d, _ in good_data]
        bad_data = [(d, False) for d, _ in bad_data]
        data = good_data + bad_data
        np.random.RandomState(0).shuffle(data)
        for seq, judgment in data:
            cost = self.hypothesis.cost(seq)
            scored.append((cost, seq, judgment))
        scored = sorted(scored, key=lambda t: t[0])
        best_acc = 0
        for i in range(len(scored)):
            cost, seq, judgment = scored[i]
            tp = len([j for c, s, j in scored[:i] if j])
            tn = len([j for c, s, j in scored[i:] if not j])
            acc = (tp + tn) / len(scored)
            best_acc = max(best_acc, acc)
        return best_acc

    def top_features(self):
        feats = []
        hyp = self.hypotheses[0]
        for i, ngram_feat in enumerate(hyp.ngram_features.keys()):
            parts = " :: ".join(hyp.feature_vocab.get_rev(f) for f in ngram_feat)
            feats.append((hyp.probs[i].item(), parts))
        return sorted(feats)[-5:]

    def all_features(self):
        feats = []
        for i, ngram_feat in enumerate(self.hypothesis.ngram_features.keys()):
            parts = " :: ".join(self.hypothesis.feature_vocab.get_rev(f) for f in ngram_feat)
            feats.append((self.hypothesis.probs[i].item(), parts))
        return sorted(feats)

    def propose(self, n_candidates, train_candidate):
        obs_set = set(s for s, j in self.observations)
        if self.strategy == "train":
            return train_candidate

        candidates = []
        for _ in range(n_candidates * 10):
            candidate = self.dataset.random_seq()
            if candidate in obs_set:
                continue
            candidates.append(candidate)
            if len(candidates) == n_candidates:
                break

        if self.strategy == "unif":
            return candidates[0]
        
        if self.strategy == "entropy":
            scored = [(self.hypothesis.entropy(c), c) for c in candidates]
            return max(scored)[0]

        assert False
