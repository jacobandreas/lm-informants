from datasets import BOUNDARY, Vocab
import util

import copy
import itertools as it
import numpy as np
from scipy.special import logsumexp
import re

import torch
from torch import nn


class BilinearScorer:
    def __init__(self, vocab, dim, embeddings=None, weights=None):
        self.vocab = vocab
        self.dim = dim
        if embeddings is None:
            self.embeddings = np.random.random(size=(len(vocab), dim)) - 0.5
        else:
            self.embeddings = embeddings
        if weights is None:
            self.weights = np.random.random(size=(dim, dim)) - 0.5
        else:
            self.weights = weights

    def score(self, seq):
        costs = []
        for (t1, t2) in util.ngrams(seq, 2):
            e1 = self.embeddings[t1]
            e2 = self.embeddings[t2]
            cost = e1.T @ self.weights @ e2
            costs.append(cost)
        return np.mean(costs)

    def perturb(self, seq):
        new_embeddings = self.embeddings.copy()
        new_weights = self.weights.copy()
        for ngram in util.ngrams(seq, 2):
            for t in ngram:
                new_embeddings[t] += (np.random.random(size=self.dim) - 0.5) * 0.05
        new_weights += (np.random.random(size=(self.dim, self.dim)) - 0.5) * 0.05
        new_embeddings = np.clip(new_embeddings, -2, 2)
        new_weights = np.clip(new_weights, -2, 2)
        return BilinearScorer(self.vocab, self.dim, new_embeddings, new_weights)


class MeanFieldScorer:  # this is us
    def __init__(self, dataset):
        self.ORDER = 3
<<<<<<< HEAD
        #        self.LOG_LOG_ALPHA_RATIO = 45 # 45 is what Jacob set # was 500
        #        alpha = 0.9999999999999999
        #        self.LOG_LOG_ALPHA_RATIO = np.log(np.log(alpha/(1-alpha))) # 45 is what Jacob set # was 500
=======
#        self.LOG_LOG_ALPHA_RATIO = 45 # 45 is what Jacob set # was 500
#        alpha = 0.9999999999999999
#        self.LOG_LOG_ALPHA_RATIO = np.log(np.log(alpha/(1-alpha))) # 45 is what Jacob set # was 500
>>>>>>> 1564f14 (increment)
        self.LOG_LOG_ALPHA_RATIO = .1
        print("log log alpha ratio: ", self.LOG_LOG_ALPHA_RATIO)
        self.dataset = dataset
        self.phoneme_features, self.feature_vocab = _load_phoneme_features(dataset)
        self.ngram_features = {}
        for ff in it.product(range(len(self.feature_vocab)), repeat=self.ORDER):
            self.ngram_features[ff] = len(self.ngram_features)
        self.probs = 0.2 * np.ones(len(self.ngram_features))

<<<<<<< HEAD
    def update_batch(self, features_seen_index, verbose=True):
        features_with_info = features_seen_index.keys()
        # print(features,"are the features in ",seq)
        constraint_probs = self.probs[features_with_info]
        # constraint_probs = [constraint_prob for constraint_prob in self.probs.keys() if constraint_prob in features_with_info]# error here; this is expecting a vector of bools or sth, but what I'm working with is a list of features
        # new_probs = self.probs.copy()
        new_probs = copy.deepcopy(self.probs)
        clip_val = 10

        for i in range(len(features_with_info)):
            this_prob = constraint_probs[i]
            other_probs = np.concatenate([constraint_probs[:i], constraint_probs[i + 1:]])
            log_p_others = np.log(1 - other_probs).sum()
            p_others = (1 - other_probs).prod()
            log_score = (
                    np.log(this_prob) - np.log(1 - this_prob)
                    + (features_seen_index[features_with_info[i]]) * (
                        np.exp(min(log_p_others + self.LOG_LOG_ALPHA_RATIO, clip_val)))
                #                    + p_others * np.exp(self.LOG_LOG_ALPHA_RATIO)
            )
            log_score = np.clip(log_score, -clip_val, clip_val)

            posterior = 1 / (1 + np.exp(-log_score))
            new_probs[features_with_info[i]] = posterior = np.clip(posterior, 0.000001, 0.999999)
            if verbose:
                new_prob = new_probs[features_with_info[i]]
                change = new_prob - this_prob
                if abs(change) > 0.0001:
                    print(
                        f"feat: {i}, before: {this_prob.round(3)}, after: {(new_prob).round(3)} ({(change).round(3)})")
                    print(f" | before sigmoid: {log_score}")
                    print(f" | prior terms: {np.log(this_prob) - np.log(1 - this_prob)}")
                    print(f" | p others: {p_others}")
                    print(f" | log p others: {log_p_others}")
                    print(f" | log alpha: {np.exp(self.LOG_LOG_ALPHA_RATIO)}")
                    print(f" | m term unclipped: {np.exp((log_p_others + self.LOG_LOG_ALPHA_RATIO))}")
                    print(f" | m term: {np.exp(min(log_p_others + self.LOG_LOG_ALPHA_RATIO, clip_val))}")

        # print("extrema", new_probs.max(), new_probs.min(), new_probs.mean())

        self.probs = new_probs
        # sign = int(judgment) * -1  # add if judgment is False, subtract if True
        # update_unclipped = sign * np.exp(min(log_p_others + self.LOG_LOG_ALPHA_RATIO, clip_val))
        # update_clipped = sign * np.exp(log_p_others + self.LOG_LOG_ALPHA_RATIO)
        results = {
            "new_probs": new_probs,
            "log_p_others": log_p_others,
            "update_clipped": log_score, }
        return results

    def update_one_step(self, seq, judgment, verbose=True):  # was originally called update
        features = self._featurize(seq).nonzero()[0]
        # print(features,"are the features in ",seq)
        constraint_probs = self.probs[features]
        # new_probs = self.probs.copy()
        new_probs = copy.deepcopy(self.probs)
        clip_val = 10
=======
    def update_one_step(self, seq, judgment, verbose=True): # was originally called update
        features = self._featurize(seq).nonzero()[0]
        #print(features,"are the features in ",seq)
        constraint_probs = self.probs[features]
        #new_probs = self.probs.copy()
        new_probs = copy.deepcopy(self.probs)
        clip_val = 10 
>>>>>>> 1564f14 (increment)

        for i in range(len(features)):
            this_prob = constraint_probs[i]
            other_probs = np.concatenate([constraint_probs[:i], constraint_probs[i + 1:]])
            log_p_others = np.log(1 - other_probs).sum()
            p_others = (1 - other_probs).prod()
            if judgment:
                # frac satisfying if off is prob that the others are all off
                # frac satisfying if on is 0
                log_score = (
                        np.log(this_prob) - np.log(1 - this_prob)
                        - np.exp(min(log_p_others + self.LOG_LOG_ALPHA_RATIO, clip_val))
                    #                    - p_others * np.exp(self.LOG_LOG_ALPHA_RATIO)
                )
                # log_score = -10000

            else:
                # frac satifying if off is prob that any other is on
                # frac satisfying if on is 1
                log_score = (
                        np.log(this_prob) - np.log(1 - this_prob)
                        + np.exp(min(log_p_others + self.LOG_LOG_ALPHA_RATIO, clip_val))
                    #                    + p_others * np.exp(self.LOG_LOG_ALPHA_RATIO)
                )
            log_score = np.clip(log_score, -clip_val, clip_val)

            posterior = 1 / (1 + np.exp(-log_score))
            new_probs[features[i]] = posterior = np.clip(posterior, 0.000001, 0.999999)
            if verbose:
                new_prob = new_probs[features[i]]
                change = new_prob - this_prob
                if abs(change) > 0.0001:
                    print(
                        f"feat: {i}, before: {this_prob.round(3)}, after: {(new_prob).round(3)} ({(change).round(3)})")
                    print(f" | before sigmoid: {log_score}")
                    print(f" | prior terms: {np.log(this_prob) - np.log(1 - this_prob)}")
                    print(f" | p others: {p_others}")
                    print(f" | log p others: {log_p_others}")
                    print(f" | log alpha: {np.exp(self.LOG_LOG_ALPHA_RATIO)}")
                    print(f" | m term unclipped: {np.exp((log_p_others + self.LOG_LOG_ALPHA_RATIO))}")
                    print(f" | m term: {np.exp(min(log_p_others + self.LOG_LOG_ALPHA_RATIO, clip_val))}")

        # print("extrema", new_probs.max(), new_probs.min(), new_probs.mean())

        self.probs = new_probs
        sign = int(judgment) * -1  # add if judgment is False, subtract if True
        update_unclipped = sign * np.exp(min(log_p_others + self.LOG_LOG_ALPHA_RATIO, clip_val))
        update_clipped = sign * np.exp(log_p_others + self.LOG_LOG_ALPHA_RATIO)
        results = {
            "new_probs": new_probs,
            "log_p_others": log_p_others,
            "update_unclipped": update_unclipped,
            "update_clipped": update_clipped, }
        return results

    def update(self, seq, judgment, verbose=True):
        results = []

        orig_probs = copy.deepcopy(self.probs)

        if verbose:
            features = self._featurize(seq).nonzero()[0]
            print(f"seq: {seq}")
            print(f"featurized: {features}")
            print(f"decoded seq: {self.dataset.vocab.decode(seq)}")
            print(f"Judgment: {judgment}")
        #            print(f"Probs before updating: {orig_probs}")
        #   print("updating!")
        target_item = False
        if not target_item:
            old_probs = copy.deepcopy(self.probs)
            step_results = self.update_one_step(seq, judgment, verbose=verbose)
            results.append(step_results)
            new_probs = step_results["new_probs"]
            difference_vector = np.subtract(new_probs, old_probs)
        else:
            old_cost = copy.deepcopy(self.logprob(seq, judgment))
            step_results = self.update_one_step(seq, judgment, verbose=verbose)
            results.append(step_results)
            new_probs = step_results["new_probs"]
            new_cost = self.logprob(seq, judgment)

            difference_vector = np.subtract(new_cost, old_cost)
            # print(old_cost,new_cost,"cost thing",difference_vector)

        num_updates = 1

        error = abs(difference_vector).sum()
        if verbose:
            print("error: ", error)
        # print(error)
        tolerance = 0.001
        if judgment == True or judgment == False:  # symmetric update; if you want asymmetric, comment out after true
            #        if judgment == True:
            #        if False:
            while error > tolerance:
                if not target_item:
<<<<<<< HEAD
                    # print(error)
                    # print(old_probs)
=======
                #print(error)
                #print(old_probs)
>>>>>>> 1564f14 (increment)
                    old_probs = copy.deepcopy(new_probs)
                    step_results = self.update_one_step(seq, judgment, verbose=verbose)
                    new_probs = step_results["new_probs"]
                    results.append(step_results)
                    difference_vector = np.absolute(np.subtract(new_probs, old_probs))
                    error = abs(difference_vector).sum()
                    # print(error)
                    # print(new_probs)
                else:
                    old_cost = copy.deepcopy(new_cost)
                    step_results = self.update_one_step(seq, judgment, verbose=verbose)
                    new_probs = step_results["new_probs"]
                    results.append(step_results)
                    new_cost = self.logprob(seq, judgment)

                    difference_vector = np.subtract(new_cost, old_cost)
                    error = abs(difference_vector).sum()
                    # print(old_cost, new_cost, "cost thing", difference_vector)

                num_updates += 1

        # print("converged!",error)
        if verbose:
            #            print(f"Probs after updating: {new_probs}")
            feature_prob_changes = new_probs[features] - orig_probs[features]
            print(f"Update in probs of features in seq (new-orig): \n{(feature_prob_changes).round(5)}")
            print(f"Num updates: {num_updates}")
        return new_probs, results

        #
        # features = self._featurize(seq).nonzero()[0]
        # # print(features,"are the features in ",seq)
        # constraint_probs = self.probs[features]
        # new_probs = self.probs.copy()
        #
        # for i in range(len(features)):
        #     this_prob = constraint_probs[i]
        #     other_probs = np.concatenate([constraint_probs[:i], constraint_probs[i + 1:]])
        #     log_p_others = np.log(1 - other_probs).sum()
        #     if judgment:
        #         # frac satisfying if off is prob that the others are all off
        #         # frac satisfying if on is 0
        #         log_score = (
        #                 np.log(this_prob) - np.log(1 - this_prob)
        #                 - np.exp(min(log_p_others + self.LOG_LOG_ALPHA_RATIO, 10))
        #         )
        #
        #     else:
        #         # frac satifying if off is prob that any other is on
        #         # frac satisfying if on is 1
        #         log_score = (
        #                 np.log(this_prob) - np.log(1 - this_prob)
        #                 + np.exp(min(log_p_others + self.LOG_LOG_ALPHA_RATIO, 10))
        #         )
        #     log_score = np.clip(log_score, -3, 3)
        #
        #     posterior = 1 / (1 + np.exp(-log_score))
        #     new_probs[features[i]] = posterior = np.clip(posterior, 0.01, 0.99)
        #
        # # print("extrema", new_probs.max(), new_probs.min(), new_probs.mean())
        #
        # self.probs = new_probs
        #
        # return new_probs

    def _featurize(self, seq):  # Canaan edit to do long distance
        features = np.zeros(len(self.ngram_features))
        for i in range(len(seq) - self.ORDER + 1):
            features_here = [self.phoneme_features[seq[j]].nonzero()[0] for j in range(i, i + self.ORDER)]
            for ff in it.product(*features_here):
                features[self.ngram_features[ff]] += 1
        return features

    def cost(self, seq):
        return -self.logprob(seq, True)

    def logprob(self, seq, judgment, length_norm=False):
        features = self._featurize(seq).nonzero()[0]
        num_features_active = len(features)
        constraint_probs = self.probs[features]

        logprob_ok = np.log(1 - constraint_probs).sum()
        if judgment:
            if length_norm:
                # print("ok!")
                # print(logprob_ok/num_features_active)
                return (logprob_ok / num_features_active)
            else:
                return logprob_ok
        else:
            if length_norm:
                return np.log1p(-torch.exp(logprob_ok)) / num_features_active
            else:
                # print(type(logprob_ok))
                logprob_ok = torch.from_numpy(np.array(logprob_ok))
                # print(type(logprob_ok))

                return np.log1p(-torch.exp(logprob_ok))

    def entropy(self, seq, debug=False, length_norm=False):
        features = self._featurize(seq).nonzero()[0]
        constraint_probs = self.probs[features]
        feat_entropies = (
                -constraint_probs * np.log(constraint_probs)
                - (1 - constraint_probs) * np.log(1 - constraint_probs)
        )
        if length_norm:
            return feat_entropies.sum() / len(constraint_probs)
        else:
            return feat_entropies.sum()

    def entropy_pred(self, seq):
        # cost is -log_prob, c is log_prob
        c = -1 * self.cost(seq)
        # p is prob
        p = np.exp(c)
        ent = -p * np.log(p) - ((1 - p) * np.log(1 - p))
        return ent


class LogisticSeqScorer(nn.Module):
    _feature_cache = {}
    _phoneme_feature_cache = {}

    def __init__(self, dataset):
        super().__init__()

        self.ORDER = 2

        self.dataset = dataset
        self.phoneme_features, self.feature_vocab = _load_phoneme_features(dataset)
        self.ngram_features = {}
        for ff in it.product(range(len(self.feature_vocab)), repeat=self.ORDER):
            self.ngram_features[ff] = len(self.ngram_features)

        self.prior = torch.rand(len(self.ngram_features))
        self.params = nn.Parameter(self.prior.clone())

    def cost_part(self, feats):
        pos_params = self.params ** 2
        if len(feats.shape) == 1:
            return -torch.dot(pos_params, feats)
        elif len(feats.shape) == 3:
            pos_params = pos_params[None, None, :]
            return -(pos_params * feats).sum(dim=2)
        else:
            assert False

    # @profile
    def cost(self, seq, grad=False):
        features, alt_features = self._featurize(seq)
        neg_cost = 0
        with torch.set_grad_enabled(grad):
            for i in range(len(features)):
                score_here = self.cost_part(features[i])
                alt_scores_here = torch.stack([self.cost_part(f) for f in alt_features[i]])
                neg_cost += score_here - torch.logsumexp(alt_scores_here, dim=0)

        assert neg_cost < 0
        if grad:
            return -neg_cost
        else:
            return -neg_cost.item()

    def loss(self, dataset):
        numerators_good = torch.zeros(len(self.ngram_features))
        denominators_good = []
        numerators_bad = []
        denominators_bad = []
        for seq, judgment in dataset:
            features, alt_features = self._featurize(seq)
            nums_bad_here = 0
            denoms_bad_here = []
            for i in range(len(features)):
                if judgment:
                    numerators_good += features[i]
                    denominators_good.append(torch.stack(alt_features[i]))
                else:
                    nums_bad_here += features[i]
                    denoms_bad_here.append(torch.stack(alt_features[i]))
            if len(denoms_bad_here) > 0:
                numerators_bad.append(nums_bad_here)
                denominators_bad.append(denoms_bad_here)
        loss_good = 0
        if len(denominators_good) > 0:
            denominators_good = torch.stack(denominators_good)
            ll_good = (
                    self.cost_part(numerators_good)
                    - ((self.cost_part(denominators_good)).logsumexp(dim=1).sum())
            )
            loss_good = -ll_good

        loss_bad = 0
        if len(denominators_bad) > 0:
            for i in range(len(denominators_bad)):
                nums_bad_here = numerators_bad[i]
                denoms_bad_here = torch.stack(denominators_bad[i])
                ex_logprob = (
                        self.cost_part(nums_bad_here)
                        - (self.cost_part(denoms_bad_here).logsumexp(dim=1).sum())
                )
                loss = -torch.log1p(-torch.exp(ex_logprob / 1000))
                loss_bad += loss

        assert loss_good >= 0
        assert loss_bad >= 0

        reg = ((self.params - self.prior) ** 2).sum()

        return (loss_good + loss_bad) / len(dataset) + 0.1 * reg

    def _featurize(self, seq):
        if seq in LogisticSeqScorer._feature_cache:
            return LogisticSeqScorer._feature_cache[seq]
        features = []
        alt_features = []

        for i in range(len(seq)):
            ctx = seq[max(i - self.ORDER + 1, 0):i]
            while len(ctx) < self.ORDER - 1:
                ctx = (self.dataset.vocab.get(BOUNDARY),) + ctx
            tgt = seq[i]
            if (ctx, tgt) in LogisticSeqScorer._phoneme_feature_cache:
                f, af = LogisticSeqScorer._phoneme_feature_cache[ctx, tgt]
                features.append(f)
                alt_features.append(af)
                continue
            f = torch.zeros(len(self.ngram_features))
            af = [torch.zeros(len(self.ngram_features)) for _ in range(len(self.phoneme_features))]
            ctx_features = [self.phoneme_features[p].nonzero()[0] for p in ctx]
            tgt_features = self.phoneme_features[tgt].nonzero()[0]
            conjunctions = (it.product(*(ctx_features + [tgt_features])))
            for conj in conjunctions:
                f[self.ngram_features[conj]] += 1
            for alt_tgt in range(len(self.phoneme_features)):
                if alt_tgt == 0 and self.dataset.onset:
                    continue
                alt_tgt_features = self.phoneme_features[alt_tgt].nonzero()[0]
                alt_conjunctions = (it.product(*(ctx_features + [alt_tgt_features])))
                for alt_conj in alt_conjunctions:
                    af[alt_tgt][self.ngram_features[alt_conj]] += 1
            LogisticSeqScorer._phoneme_feature_cache[ctx, tgt] = (f, af)
            features.append(f)
            alt_features.append(af)

        LogisticSeqScorer._feature_cache[seq] = (features, alt_features)
        return features, alt_features


class LogisticScorer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.phoneme_features, self.feature_vocab = _load_phoneme_features(dataset)
        self.ngram_features = {}
        for f1 in range(len(self.feature_vocab)):
            for f2 in range(len(self.feature_vocab)):
                self.ngram_features[f1, f2] = len(self.ngram_features)
        self.params = 0.00 * (np.random.random(len(self.ngram_features)) - 0.5)

    def cost(self, seq):
        feats = self._featurize(seq)
        cost = np.dot(self.params, feats)
        return cost

    def grad(self, seq, judgment):
        feats = self._featurize(seq)
        cost = np.dot(self.params, feats)
        prob = 1 / (1 + np.exp(cost))
        grad = -(int(judgment) - prob) * feats
        return cost, grad

    # TODO cache this
    def _featurize(self, seq):
        features = np.zeros(len(self.ngram_features))
        for i in range(len(seq) - 1):
            for f1 in np.nonzero(self.phoneme_features[seq[i]])[0]:
                for f2 in np.nonzero(self.phoneme_features[seq[i + 1]])[0]:
                    features[self.ngram_features[f1, f2]] += 1
        return features


class MLPScorer:
    def __init__(self, vocab, dim, params=None):
        self.vocab = vocab
        self.dim = dim
        if params is None:
            self.embeddings = np.random.random(size=(len(vocab), dim)) - 0.5
            self.weights1 = np.random.random(size=(3 * dim, dim)) - 0.5
            self.weights2 = np.random.random(size=(dim, 1)) - 0.5
        else:
            self.embeddings = params["embeddings"]
            self.weights1 = params["weights1"]
            self.weights2 = params["weights2"]

    def score(self, seq):
        costs = []
        for (t1, t2, t3) in util.ngrams(seq, 3):
            e1 = self.embeddings[t1]
            e2 = self.embeddings[t2]
            e3 = self.embeddings[t3]
            e = np.concatenate((e1, e2, e3))
            # h = util.sigmoid(e @ self.weights1)
            h = np.tanh(e @ self.weights1)
            o = h @ self.weights2
            costs.append(o)
        return np.mean(costs)

    def perturb(self, seq):
        new_embeddings = self.embeddings.copy()
        new_weights1 = self.weights1.copy()
        new_weights2 = self.weights2.copy()
        for ngram in util.ngrams(seq, 2):
            for t in ngram:
                new_embeddings[t] += (np.random.random(size=self.dim) - 0.5) * 0.05
        new_weights1 += (np.random.random(size=(3 * self.dim, self.dim)) - 0.5) * 0.05
        new_weights2 += (np.random.random(size=(self.dim, 1)) - 0.5) * 0.05
        new_embeddings = np.clip(new_embeddings, -2, 2)
        new_weights1 = np.clip(new_weights1, -2, 2)
        new_weights2 = np.clip(new_weights2, -2, 2)
        return MLPScorer(self.vocab, self.dim, params={
            "embeddings": new_embeddings,
            "weights1": new_weights1,
            "weights2": new_weights2
        })

    def penalty(self):
        return 0


class VotingScorer:
    def __init__(self, members):
        self.members = members

    def cost(self, seq, debug=False):
        mcosts = [m.cost(seq) for m in self.members]
        num_zero = len([c for c in mcosts if c == 0])
        if num_zero >= len(self.members) // 2:
            return 0
        return np.mean(mcosts)


class HWScorer:
    def __init__(self, dataset, ngram_features=None):
        self.dataset = dataset

        self.phoneme_features, self.feature_vocab = _load_phoneme_features(dataset)
        if ngram_features is None:
            self.ngram_features = _load_ngram_features(self.feature_vocab)
        else:
            self.ngram_features = ngram_features

    def applicable_features(self, seq):
        for order in (2, 3):
            for i in range(len(seq) - order + 1):
                for ngram_feature, feature_cost in self.ngram_features[order]:
                    success = True
                    for j in range(order):
                        phoneme_features = self.phoneme_features[seq[i + j]]
                        success &= all(phoneme_features[f] for f in ngram_feature[j])
                    if success:
                        yield (ngram_feature, feature_cost)

    def cost(self, seq, debug=False):
        seq_cost = 0
        for ngram_feature, feature_cost in self.applicable_features(seq):
            if debug:
                print(self.pp_feature(ngram_feature))
            seq_cost += feature_cost
        return seq_cost

    def add(self, feature, cost):
        new_features = copy.deepcopy(self.ngram_features)
        new_features[len(new_features)].append((feature, cost))
        return HWScorer(self.dataset, new_features)

    def pp_feature(self, ngram_feature):
        out = []
        for j in range(len(ngram_feature)):
            part = []
            for f in ngram_feature[j]:
                part.append(self.feature_vocab.get_rev(f))
            out.append(f"[{', '.join(part)}]")
        return " ".join(out)

    def pp(self, gold={}):
        out = []
        for order in [2, 3]:
            for feature, cost in self.ngram_features[order]:
                out.append(self.pp_feature(feature))
                if feature in gold:
                    out[-1] += " *"
        return "\n" + "\n".join(out)

    def penalty(self):
        return len(self.ngram_features[2]) + len(self.ngram_features[3])

    def perturb(self, seq):
        feature_pairs = set()
        for t1, t2 in util.ngrams(seq, 2):
            feats1, = self.phoneme_features[t1].nonzero()
            feats2, = self.phoneme_features[t2].nonzero()
            feature_pairs |= set(it.product(feats1, feats2))
        feature_pairs = sorted(feature_pairs)
        random_pair = feature_pairs[np.random.randint(len(feature_pairs))]

        applicable_features = list(self.applicable_features(seq))
        if len(applicable_features) == 0:
            random_applicable = None
        else:
            random_applicable = applicable_features[np.random.randint(len(applicable_features))]

        new_features = copy.deepcopy(self.ngram_features)

        def remove(feature):
            existing_set = new_features[len(feature[0])]
            new_set = [f for f in existing_set if f != feature]
            assert len(new_set) < len(existing_set)
            new_features[len(feature)] = new_set

        action = np.random.randint(4)
        if action == 0 or random_applicable is None:  # create
            # TODO might already be there
            f1, f2 = random_pair
            new_feature = ((f1,), (f2,))
            new_features[2].append((new_feature, -1.))

        elif action == 1:  # insert
            new_feature, new_cost = random_applicable
            remove(random_applicable)
            offset = np.random.choice((0, 1))
            new_feature = list(list(t) for t in new_feature)
            new_feature[offset].append(random_pair[0])
            if offset + 1 >= len(new_feature):
                new_feature.append([])
            new_feature[offset + 1].append(random_pair[1])
            new_feature = tuple(tuple(l) for l in new_feature)
            new_features[len(new_feature)].append((new_feature, new_cost))

        elif action == 2:  # delete
            remove(random_applicable)

        elif action == 3:  # reweight
            remove(random_applicable)
            new_feature, new_cost = random_applicable
            new_cost *= np.exp(np.random.random() - 0.5)
            new_features[len(new_feature)].append((new_feature, new_cost))

        # print(seq)
        # print(self.ngram_features)
        # print(new_features)
        # print()

        new = HWScorer(self.dataset, new_features)
        # if new.penalty() < self.penalty():
        #    print(self.penalty(), new.penalty())
        return new


def _load_phoneme_features(dataset):
    phoneme_features = {}
    feature_vocab = Vocab()

    with open("data/hw/atr_harmony_features.txt") as reader:
        header = next(reader)
        feat_names = header.strip().split("\t")
        feat_names.append("word_boundary")
        for feat_name in feat_names:
            feature_vocab.add(f"+{feat_name}")
            feature_vocab.add(f"-{feat_name}")
        for line in reader:
            line = line.strip().split("\t")
            phoneme = line[0]
            feat_vals = line[1:]
            feat_vec = np.zeros(len(feature_vocab))
            for i, feat_val in enumerate(feat_vals):
                if feat_val == "0":
                    continue
                feat = feature_vocab.get(f"{feat_val}{feat_names[i]}")
                feat_vec[feat] = 1
            # feat_vec[feature_vocab.get("-word_boundary")] = 1
            phoneme_features[dataset.vocab.get(phoneme)] = feat_vec

        boundary_vec = np.zeros(len(feature_vocab))
        boundary_vec[feature_vocab.get("+word_boundary")] = 1
        phoneme_features[dataset.vocab.get(BOUNDARY)] = boundary_vec

    return phoneme_features, feature_vocab


def _load_ngram_features(feature_vocab):
    ngram_features = {2: [], 3: []}
    with open("data/hw/atr_feature_weights.txt") as reader:
        for line in reader:
            line = line.strip().split("\t")
            features = re.findall(r"\[([^\[\]]+)\]", line[0])
            feature_template = []
            for feature in features:
                feature_template_part = []
                parts = feature.split(",")
                for part in parts:
                    feature_template_part.append(feature_vocab.get(part))
                feature_template.append(tuple(feature_template_part))
            feature_template = tuple(feature_template)
            weight = float(line[-1])
            order = len(feature_template)
            ngram_features[order].append((feature_template, weight))
    return ngram_features