from collections import Counter
from functools import reduce
import itertools as it
import numpy as np
import scorers
from tqdm import tqdm
from scipy.special import logsumexp

import torch
from torch import nn, optim

class Learner:
    def __init__(self, dataset, strategy, linear_train_dataset, index_of_next_item):
        self.dataset = dataset
        self.linear_train_dataset = linear_train_dataset
        self.index_of_next_item = index_of_next_item
        #self.next_token_button = next_token_button

        self.propose_train = 0
        if strategy == "diff":
            #self.strategy = lambda l: max(l) - min(l)
            def fn(costs):
                #probs = [1/(1+np.exp(c)) for c in costs]
                probs = costs
                return max(probs) - min(probs)
            self.strategy = fn
        elif strategy == "max":
            def fn(costs):
                #probs = [1/(1+np.exp(c)) for c in costs]
                return -np.mean(costs)
            self.strategy = fn
        elif strategy == "unif":
            def fn(costs):
                assert False, "unreachable"
                return 0
            self.strategy = fn
        elif strategy == "std":
            def fn(costs):
                #probs = [1/(1+np.exp(c)) for c in costs]
                #probs = costs
                #probs = [c / torch.exp
                #import ipdb; ipdb.set_trace()
                #probs = [np.exp(-c) for c in costs]
                probs = costs
                #if np.std(probs) > 0:
                #    import ipdb; ipdb.set_trace()
                return np.std(probs)
            self.strategy = fn
        elif strategy == "interleave":
            def fn(costs):
                return np.std(costs)
            self.strategy = fn
            self.propose_train = 0.5
        elif strategy == "train":
            def fn(costs):
                return 0
            self.strategy = fn
            self.propose_train = 1
        elif strategy == "entropy":
            pass
        elif strategy == "eig":
            pass
        else:
            assert False

        self.strategy_name = strategy

        self.hypotheses = None
        self.observations = None

    def initialize(self, n_hyps):
        self.hypotheses = [
            self.initialize_hyp() for _ in range(n_hyps)
        ]
        self.observations = []


    def propose(self, n_candidates=100):
        obs_set = set(s for s, j in self.observations)
        if np.random.random() < self.propose_train:
            while True:
                #seq = self.dataset.random_example()
                seq = self.linear_train_dataset[self.index_of_next_item]
                #print("proposing item",seq,"with index",self.index_of_next_item)

                self.index_of_next_item += 1
                if seq not in obs_set:
                    return seq
        candidates = []
        while len(candidates) == 0:
            candidates = [self.dataset.random_seq() for _ in range(n_candidates)]
            candidates = [c for c in candidates if c not in obs_set]
        scores = [
            [h.cost(c) for h in self.hypotheses]
            for c in candidates
        ]
        assert self.strategy_name == "entropy"
        scored_candidates = list(zip(candidates, scores))
        best = max(scored_candidates, key=lambda p: p[1])
        return best[0]


class VBLearner(Learner):
    def __init__(self, dataset, strategy, linear_train_dataset,index_of_next_item):
        super().__init__(dataset, strategy, linear_train_dataset, index_of_next_item )

    def initialize_hyp(self):
        return scorers.MeanFieldScorer(self.dataset)

    def observe(self, seq, judgment, update=True):
        assert len(self.hypotheses) == 1
        assert update
        self.observations.append((seq, judgment))
        self.hypotheses[0].update(seq, judgment)

    def get_eig(self, seq):
        learner_before_fussing_around = self.hypotheses[0]

        # prob of the thing being positive or negative
        prob_being_positive_a = np.exp(self.hypotheses[0].logprob(seq, True))
        prob_being_negative_a = np.exp(self.hypotheses[0].logprob(seq, False))
        prob_being_positive = prob_being_positive_a/(prob_being_positive_a+prob_being_negative_a)
        prob_being_negative = 1-prob_being_positive
        assert prob_being_positive + prob_being_negative == 1

        # entropy over features before seeing
        p = self.hypotheses[0].probs
        entropy_over_features_before_observing_item = ((p * np.log(p) + (1 - p) * np.log(1 - p)).sum())
        # entropy over features after seeing if positive

        p = self.hypotheses[0].update(seq, True)
        entropy_over_features_after_observing_item_positive = ((p * np.log(p) + (1 - p) * np.log(1 - p)).sum())
        self.hypotheses[0] = learner_before_fussing_around
        # reset learner
        # entropy over features after seeing if negative
        p = self.hypotheses[0].update(seq, False)
        entropy_over_features_after_observing_item_negative = ((p * np.log(p) + (1 - p) * np.log(1 - p)).sum())
        self.hypotheses[0] = learner_before_fussing_around
        # reset learner

        # (ent_before-ent_after_positive)*prob_positive + (ent_before-ent_after_negative)*prob_negative
        eig = (abs(entropy_over_features_before_observing_item-entropy_over_features_after_observing_item_positive)*prob_being_positive + \
        abs(entropy_over_features_before_observing_item-entropy_over_features_after_observing_item_negative)*prob_being_negative)

        #print("eig is",eig)

        return eig

    def cost(self, seq):
        return self.hypotheses[0].cost(seq)

    def full_nll(self, data):
        if len(data) == 0:
            #return np.nan
            return 0
        else:
            return sum(self.hypotheses[0].logprob(s, j) for s, j in data)

    def discriminate(self, good_data, bad_data):
        scored = []

        good_data = [(d, True) for d, _ in good_data]
        bad_data = [(d, False) for d, _ in bad_data]
        data = good_data + bad_data
        np.random.RandomState(0).shuffle(data)
        for seq, judgment in data:
            cost = self.hypotheses[0].cost(seq)
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
        hyp = self.hypotheses[0]
        for i, ngram_feat in enumerate(hyp.ngram_features.keys()):
            parts = " :: ".join(hyp.feature_vocab.get_rev(f) for f in ngram_feat)
            feats.append((hyp.probs[i].item(), parts))
        return sorted(feats)

    def propose(self, n_candidates, forbidden_data, length_norm):
        obs_set_a = set(s for s, j in self.observations)
        obs_set = set(s for s in (forbidden_data+list(obs_set_a)))
        if np.random.random() < self.propose_train:
            while True:
                seq = self.linear_train_dataset[self.index_of_next_item]
                #print("proposing item",seq,"with index",self.index_of_next_item)
                self.index_of_next_item += 1
                #seq = self.dataset.random_example()
                if seq not in obs_set:
                    return seq
        candidates = []
        while len(candidates) == 0:
            candidates = [self.dataset.random_seq() for _ in range(n_candidates)]
            candidates = [c for c in candidates if c not in obs_set]
        #import ipdb; ipdb.set_trace()
        if self.strategy_name == "entropy":
            scores = [
                self.hypotheses[0].entropy(c, length_norm=length_norm)
                for c in candidates
            ]
        elif self.strategy_name == "unif" or self.propose_train > 0:
            scores = [0 for c in candidates]
        elif self.strategy_name == "eig":
            scores = [self.get_eig(c) for c in candidates]
        else:
            raise NotImplementedError(f"strategy {self.strategy_name} not implemented")
        scored_candidates = list(zip(candidates, scores))
        best = max(scored_candidates, key=lambda p: p[1])
        #print(be, self.strategy(best[1]))
        #print(scored_candidates)
        #assert False
        #print(best[1])
        return best[0]

class LogisticLearner(Learner):
    def __init__(self, dataset, strategy):
        super().__init__(dataset, strategy)
        self.lr = 0.01
        self.reg = 0.001

    def initialize_hyp(self):
        return scorers.LogisticSeqScorer(self.dataset)
        #return scorers.LogisticScorer(self.dataset)

    def observe(self, seq, judgment, update=True):
        self.observations.append((seq, judgment))
        if update:
            new_hypotheses = [self._retrain(h) for h in self.hypotheses]
            self.hypotheses = new_hypotheses

    def top_features(self):
        feats = []
        for i, ngram_feat in enumerate(self.hypotheses[0].ngram_features.keys()):
            parts = " :: ".join(self.hypotheses[0].feature_vocab.get_rev(f) for f in ngram_feat)
            feat_scores = []
            for hyp in self.hypotheses:
                feat_scores.append(hyp.params[i].item() ** 2)
            feats.append((logsumexp(feat_scores), parts))
        return sorted(feats)[-5:]


    #@profile
    def _retrain(self, hypothesis):
        #subsample_indices = np.random.randint(
        #    len(self.observations),
        #    size=len(self.observations)
        #)
        subsampled_dataset = self.observations

        opt = optim.Adam([hypothesis.params], lr=self.lr, weight_decay=0)
        for i in tqdm(range(20)):
            loss = 0
            count = 0
            for j in range(0, len(subsampled_dataset), 32):
                batch = subsampled_dataset[j:j+32]
                batch_loss = hypothesis.loss(batch)
                opt.zero_grad()
                batch_loss.backward()
                opt.step()
                loss += batch_loss.item()
                count += 1
            #print(loss / count)
        return hypothesis

    def cost(self, seq):
        return logsumexp([h.cost(seq) for h in self.hypotheses])

    def full_nll(self, data, debug=False):
        return logsumexp([self.nll(h, data) for h in self.hypotheses])

    def discriminate(self, good_data, bad_data):
        scored = []

        good_data = [(d, True) for d, _ in good_data]
        bad_data = [(d, False) for d, _ in bad_data]
        data = good_data + bad_data
        for seq, judgment in data:
            cost = logsumexp([h.cost(seq) for h in self.hypotheses])
            scored.append((cost, seq, judgment))
        scored = sorted(scored)
        best_acc = 0
        for i in range(len(scored)):
            cost, seq, judgment = scored[i]
            #print(cost, " ".join(self.dataset.vocab.decode(seq)), judgment)
            tp = len([j for c, s, j in scored[:i] if j])
            tn = len([j for c, s, j in scored[i:] if not j])
            acc = (tp + tn) / len(scored)
            best_acc = max(best_acc, acc)
        return best_acc


    def nll(self, hypothesis, data, debug=False):
        nll = 0
        for seq, judgment in data:
            cost = hypothesis.cost(seq)
            if judgment:
                nll += cost
            else:
                nll += np.log1p(np.exp(-cost))
        if len(data) == 0:
            # return np.nan
            return 0
        else:
            assert nll > 0, nll
            return nll / len(data)

class FilterHWLearner(Learner):
    def __init__(self, dataset, strategy):
        super().__init__(dataset, strategy)

        scorer = scorers.HWScorer(self.dataset)
        self.phoneme_features = scorer.phoneme_features
        self.ngram_features = [f for f, c in scorer.ngram_features[2]]

    def initialize_hyp(self):
        return scorers.HWScorer(self.dataset, {2: [], 3: []})

    def observe(self, seq, judgment):
        self.observations.append((seq, judgment))
        new_constraints = self._enumerate_constraints(seq)
        new_hyps = [
            self._update_hyp(h, seq, judgment, new_constraints)
            for h in self.hypotheses
        ]
        #new_hyps = [h for h in new_hyps if h is not None]
        #if len(new_hyps) == 0:
        #    new_hyps = [self.initialize_hyp() for _ in self.hypotheses]
        #else:
        #    while len(new_hyps) < len(self.hypotheses):
        #        new_hyps.append(np.random.choice(new_hyps))
        self.hypotheses = new_hyps
        #print(self._consensus_hyp().pp())
        for hyp in self.hypotheses:
            print(hyp.pp(self.ngram_features))
            print()

    def _ngram_features(self, seq):
        features = set()
        for order in (2, 3):
        #for order in (2,):
            for i in range(len(seq) - order + 1):
                phonemes = seq[i:i+order]
                ph_features = [self.phoneme_features[p] for p in phonemes]
                ph_features = [f.nonzero()[0].tolist() for f in ph_features]

                ph_features_1 = [[(f,) for f in ff] for ff in ph_features]
                ph_features_2 = [list(it.combinations(ff, 2)) for ff in ph_features]
                ph_features_c = [l1 + l2 for l1, l2 in zip(ph_features_1, ph_features_2)]

                for ngram_feature in it.product(*ph_features_c):
                    #ngram_feature = tuple((f,) for f in ngram_feature)
                    features.add(tuple(ngram_feature))
        return features

    def _enumerate_constraints(self, seq):
        allowed = set()
        addable = Counter()
        for obs_seq, judgment in self.observations:
            if judgment:
                allowed |= self._ngram_features(obs_seq)
            else:
                addable.update(self._ngram_features(obs_seq))
        #print(addable)
        addable = {k for k, v in addable.items() if v >= 5}
        #print(addable)
        #print()

        constraints = (self._ngram_features(seq) & addable) - allowed
        return list(constraints)

    def _most_violated(self, hypothesis, constraints, observations):
        violations = Counter()
        for obs_seq, judgment in observations:
            if judgment:
                continue
            if hypothesis.cost(obs_seq) > 0:
                continue
            for feature in self._ngram_features(obs_seq):
                if feature in constraints:
                    violations[feature] += 1
        if len(violations) == 0:
            return []
        violations = list(violations.items())
        np.random.shuffle(violations)
        by_violations = sorted(violations, key=lambda p: p[1], reverse=True)
        worst = by_violations[0][1]
        #print(by_violations)
        #assert False

        return [k for k, v in by_violations]

        #like_worst = [k for k, v in by_violations if v == worst]
        #np.random.shuffle(like_worst)
        #return like_worst

        #worst = by_violations[-10:]
        #worst = [k for k, v in worst]
        #np.random.shuffle(worst)
        #return worst

    def _update_hyp(self, hypothesis, seq, judgment, new_constraints):
        cost = hypothesis.cost(seq)
        if (cost == 0) == (judgment):
            return hypothesis


        return self._reset_hyp_xval(hypothesis)


        if judgment:
            ok = self._check_hyp(hypothesis)
            if ok:
                #print("keeping", hypothesis.pp())
                return hypothesis
            return self._reset_hyp(hypothesis)
        else:
            #np.random.shuffle(new_constraints)
            #ordered_constraints = self._most_violated(hypothesis, new_constraints)
            #print(new_constraints)
            ordered_constraints = np.random.permutation(new_constraints)
            for constraint in ordered_constraints:
                #assert isinstance(constraint, tuple), constraint
                new_hyp = hypothesis.add(tuple(constraint), 1)
                if self._check_hyp(new_hyp):
                    #print("adding", new_hyp.pp())
                    return new_hyp
            # we got to the end without succeeding
            return self._reset_hyp(hypothesis)
            #return hypothesis

    def _check_hyp(self, hypothesis):
        #print(hypothesis.ngram_features, self.observations)
        total = 0.
        wrong = 0.
        for obs, judgment in self.observations:
            total += 1
            cost = hypothesis.cost(obs)
            if hypothesis.cost(obs) > 0 and judgment == True:
                wrong += 1
        #return wrong / total < .1 # TODO magic number
        return wrong == 0

    def _reset_hyp_xval(self, hypothesis):
        #print("reset")
        if len(self.observations) <= 2:
            return self.initialize_hyp()

        obs_perm = np.random.permutation(self.observations)
        train = obs_perm[:len(obs_perm)//2]
        val = obs_perm[len(obs_perm)//2:]

        good_feats = set()
        bad_feats = set()
        for obs, judgment in train:
            if judgment:
                good_feats |= self._ngram_features(obs)
            else:
                bad_feats |= self._ngram_features(obs)
        feats = bad_feats - good_feats
        new_hyp = self.initialize_hyp()
        nll = self.nll(new_hyp, val)
        added = 0
        while True:
            violated = self._most_violated(new_hyp, feats, train)
            if violated is None:
                break
            changed = False
            for feat in violated:
                newer_hyp = new_hyp.add(feat, 1)
                newer_nll = self.nll(newer_hyp, val)
                if newer_nll < nll:
                    #print(nll, "->", newer_nll)
                    changed = True
                    break
                break
            if changed:
                new_hyp = newer_hyp
                nll = newer_nll
                added += 1
                continue
            else:
                break
        #print("added", added, len(new_hyp.ngram_features[2]))
        return new_hyp




    def _reset_hyp(self, hypothesis):
        print("reset")
        #return self.initialize_hyp()
        ##return None

        good_feats = set()
        bad_feats = set()
        for obs, judgment in self.observations:
            if judgment:
                good_feats |= self._ngram_features(obs)
            else:
                bad_feats |= self._ngram_features(obs)

        feats = bad_feats - good_feats
        new_hyp = self.initialize_hyp()
        nll = self.nll(new_hyp, self.observations)
        #print(nll)
        while True:
            violated = self._most_violated(new_hyp, feats)
            if violated is None:
                break
            for feat in violated:
                newer_hyp = new_hyp.add(feat, 1)
                if self._check_hyp(newer_hyp):
                    #print("new hyp picked")
                    new_hyp = newer_hyp
                    break
                else:
                    import ipdb; ipdb.set_trace()
            new_nll = self.nll(new_hyp, self.observations)
            #print(new_nll)
            if new_nll >= nll:
                break
            nll = new_nll

        return new_hyp

    def _consensus_hyp(self):
        return scorers.VotingScorer(self.hypotheses)
        #all_constraints = {2: [], 3: []}
        #for order in [2]:
        #    constraints = [h.ngram_features[order] for h in self.hypotheses]
        #    constraints = reduce(lambda x, y: list(set(x) & set(y)), constraints)
        #    all_constraints[order] = constraints
        #return scorers.HWScorer(self.dataset, all_constraints)

    def full_nll(self, data, debug=False):
        consensus_hyp = self._consensus_hyp()
        return self.nll(consensus_hyp, data)
        #return np.mean([self.nll(h, data, debug) for h in self.hypotheses])

    def nll(self, hypothesis, data, debug=False):
        nll = 0
        for obs, judgment in data:
            cost = hypothesis.cost(obs)
            assert cost >= 0, cost
            if judgment:
                #nll += np.log1p(np.exp(cost))
                nll += int(cost > 0)
                if debug:
                    print(self.dataset.vocab.decode(obs))
                    hypothesis.cost(obs, debug)
                    print()
            else:
                #nll += np.log1p(np.exp(-cost))
                nll += int(cost == 0)
        if len(data) == 0:
            # return np.nan
            return 0
        assert nll >= 0, nll
        return nll / len(data)

class SimpleMHLearner(Learner):
    def __init__(self, dataset, strategy, n_hyps, dim):
        super().__init__(self, dataset, strategy)

    def initialize_hyp(self):
        #return scorers.BilinearScorer(dataset.vocab, dim)
        return scorers.MLPScorer(dataset.vocab, dim)

    def observe(self, seq, judgment):
        self.observations.append((seq, judgment))
        new_hypotheses = [self._run_mh(h, seq) for h in self.hypotheses]
        self.hypotheses = new_hypotheses

    def _run_mh(self, hypothesis, seq):
        i = 0
        orig_nll = curr_nll = self.nll(hypothesis, self.observations) + self.reg(hypothesis)
        changed = False
        while i < 100:
            proposal = hypothesis.perturb(seq)
            proposed_nll = self.nll(proposal, self.observations) + self.reg(proposal)
            #accept_crit = np.exp(50 * (curr_nll - proposed_nll))
            accept_crit = np.exp(1 * (curr_nll - proposed_nll))
            rand = np.random.random()
            if rand < accept_crit:
                hypothesis = proposal
                curr_nll = proposed_nll
                changed = True
            i += 1
        if not changed:
            print("warning: no update")
        #if curr_nll > orig_nll:
        #    print("warning: got worse", orig_nll, curr_nll)
        #else:
        #    print(orig_nll, "->", curr_nll)
        return hypothesis

    #def full_nll(self, data):
    #    return np.mean([self.nll(h, data) for h in self.hypotheses])

    #def nll(self, hypothesis, data):
    #    nll = 0
    #    for seq, judgment in data:
    #        score = hypothesis.score(obs)
    #        if judgment:
    #            nll += np.log1p(np.exp(-score))
    #        else:
    #            nll += np.log1p(np.exp(score))
    #    if len(data) == 0:
    #        return np.nan
    #    else:
    #        assert nll > 0, nll
    #        return nll / len(data)

    def reg(self, hypothesis):
        return 3 * (
            hypothesis.penalty()
            ##np.linalg.norm(hypothesis.embeddings)
            ##+ np.linalg.norm(hypothesis.weights)
            # + np.linalg.norm(hypothesis.weights1)
            # + np.linalg.norm(hypothesis.weights2)
        )
