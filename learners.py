from collections import Counter
from functools import reduce
import itertools as it
import random
import multiprocessing
import numpy as np
import scorers
from tqdm import tqdm
from scipy.special import logsumexp
from copy import deepcopy 

import torch
#from torchmetrics.functional import kl_divergence
from torch import nn, optim
from util import kl_bern, entropy

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
        elif strategy in ["entropy", "entropy_pred"]:
            pass
        elif strategy == "eig":
            pass
        elif strategy in ["eig_train_model", "eig_train_history", "kl_train_history", "kl_train_model"]:
            pass
        elif strategy == "kl":
            pass
        else:
            assert False

        self.strategy_name = strategy
        self.hypotheses = None
        self.observations = None
        
        self.chosen_strategies = []

    def initialize(
            self, n_hyps,
            log_log_alpha_ratio=1, 
            prior_prob=0.5,
            converge_type="symmetric",
            feature_type="atr_harmony",
            tolerance=0.001,
            ):
        self.hypotheses = [
            self.initialize_hyp(
                log_log_alpha_ratio=log_log_alpha_ratio, prior_prob=prior_prob, converge_type=converge_type, feature_type=feature_type, tolerance=tolerance,
                ) for _ in range(n_hyps)
        ]
        self.observations = []
        self.observed_seqs = []
        self.observed_feats = []
        self.observed_judgments = []
        self.observed_feats_unique = set()

        if self.strategy_name in [
                "eig_train_model", 
                "eig_train_history"]: 
            # track eig/kl
            metric_to_track = "entropy_diff"
        elif self.strategy_name in [
                "kl_train_model", 
                "kl_train_history"]:
            metric_to_track = "kl"
        else:
            metric_to_track = None
        
        self.metric_to_track = metric_to_track 
           
        # TODO: one of these is a dummy, eg will only populate train/eig for eig_train
        self.kls_by_strategy = {"train": [], "eig": [], "kl": []}
        self.entropy_diffs_by_strategy = {"train": [], "eig": [], "kl": []}

        """
        # this slowed things down for eig/kl because required copying large dictionaries; TODO: if don't initialize dictionaries with all features, but just ones seen, is it faster?
        # these dictionaries store batch quantitities by each feature, precomputed for efficiency in update() 
        self.batch_feats_by_feat = {f: [] for f in range(len(self.hypotheses[0].probs))}
        self.batch_other_feats_by_feat = {f: [] for f in range(len(self.hypotheses[0].probs))}
        self.batch_judgments_by_feat = {f: [] for f in range(len(self.hypotheses[0].probs))}
        """

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
    def __init__(
            self, 
            dataset, 
            strategy, 
            linear_train_dataset,index_of_next_item, 
            ):
        super().__init__(dataset, strategy, linear_train_dataset, index_of_next_item)
        self.results_by_observations = []
        self.gain_list_from_train = []
        self.gain_list_from_alterative = []
        self.strategy_for_this_candidate = None
        
    def initialize_hyp(self, log_log_alpha_ratio=1, prior_prob=0.5, converge_type="symmetric", feature_type="atr_harmony", tolerance=0.001, warm_start=False):
        return scorers.MeanFieldScorer(
                self.dataset, 
                log_log_alpha_ratio=log_log_alpha_ratio, 
                prior_prob=prior_prob,
                converge_type=converge_type,
                feature_type=feature_type,
                tolerance=tolerance,
                warm_start=warm_start,
                )

    def observe(self, seq, judgment, update=True, do_plot_wandb=False, verbose=False, batch=True):
        assert len(self.hypotheses) == 1
        assert update
        features = self.hypotheses[0]._featurize(seq).nonzero()[0]
        self.observations.append((seq, features, judgment))
        self.observed_seqs.append(seq)
        self.observed_feats.append(features)
        self.observed_judgments.append(1 if judgment else -1)
        self.observed_feats_unique.update(features)
     
        """
        # for each feature in the sequence, add current sequence to stored batch values for that feature
        for feat in features:
            self.batch_feats_by_feat[feat].append(features)
            self.batch_other_feats_by_feat[feat].append([f for f in features if f != feat])
            self.batch_judgments_by_feat[feat].append(judgment) 
        """

#        _, results = self.hypotheses[0].update(seq, judgment)
        if batch:
            seqs_to_observe = self.observations
            ordered_feats = self.observed_feats
            ordered_judgments = self.observed_judgments
            ordered_seqs = self.observed_seqs
            feats_to_update = list(self.observed_feats_unique)
        
            """
            batch_other_feats_by_feat = self.batch_other_feats_by_feat
            batch_feats_by_feat = self.batch_feats_by_feat
            batch_judgments_by_feat = self.batch_judgments_by_feat
            """
            
        else:
            seqs_to_observe = [self.observations[-1]]
            ordered_feats = [self.observed_feats[-1]]
            ordered_judgments = [self.observed_judgments[-1]]
            ordered_seqs = [self.observed_seqs[-1]]
            feats_to_update = list(set(ordered_feats))

            """
            # TODO: make sure this is robust, we basically only look at this last sequence
            batch_feats_by_feat = {feat: features for feat in features}
            batch_other_feats_by_feat = {feat: [f for f in features if f != feat] for feat in features}
            batch_judgments_by_feat = {feat: judgment for feat in features}
            """

        # TODO: redundant with some calculations in main.py; remove there?
        probs_before = self.hypotheses[0].probs.copy()

        _, results = self.hypotheses[0].update(
                ordered_feats, ordered_judgments, 
                verbose=verbose, do_plot_wandb=do_plot_wandb, 
                feats_to_update=feats_to_update)
        self.results_by_observations.append(results)

        entropy_before = entropy(probs_before)
        entropy_after = entropy(self.hypotheses[0].probs)
        entropy_diff = entropy_before - entropy_after
        kl = kl_bern(self.hypotheses[0].probs.copy(), probs_before).sum()
      
        # TODO: hacky; this assumes that observe() is always called after propose(), bc self.chosen_strategies is appended to when a candidate is proposed
        chosen_strategy = self.chosen_strategies[-1]
        if chosen_strategy in self.kls_by_strategy:
            self.kls_by_strategy[chosen_strategy].append(kl)
            self.entropy_diffs_by_strategy[chosen_strategy].append(entropy_diff)
        else:
            self.kls_by_strategy[chosen_strategy] = [kl]
            self.entropy_diffs_by_strategy[chosen_strategy] = [entropy_diff]

    # TODO: only consider features in sequence as in scorer.entropy()? (Should be equivalent bc probs for features not in seq won't change?)
    def get_eig(self, seq):
        orig_probs = deepcopy(self.hypotheses[0].probs)
#        print("learner before fussing around:", learner_before_fussing_around.probs)

        # prob of the thing being positive or negative
        prob_being_positive_a = np.exp(self.hypotheses[0].logprob(seq, True))
        prob_being_negative_a = np.exp(self.hypotheses[0].logprob(seq, False))
        prob_being_positive = prob_being_positive_a/(prob_being_positive_a+prob_being_negative_a)
        prob_being_negative = 1-prob_being_positive
        assert prob_being_positive + prob_being_negative == 1

        # entropy over features before seeing
        p = self.hypotheses[0].probs
        # TODO: confirm -1 after sum
        entropy_over_features_before_observing_item = -1 * ((p * np.log(p) + (1 - p) * np.log(1 - p))).sum()
        test_entropy_over_features_before_observing_item = (-1 * (p * np.log(p) + (1 - p) * np.log(1 - p))).sum()
        assert entropy_over_features_before_observing_item > 0
        assert entropy_over_features_before_observing_item == test_entropy_over_features_before_observing_item
    

        # entropy over features after seeing if positive

#        p, _ = self.hypotheses[0].update(seq, True, verbose=False)
        features = self.hypotheses[0]._featurize(seq).nonzero()[0]
        
        p, results = self.hypotheses[0].update(self.observed_feats+[features], self.observed_judgments+[True], verbose=False, )
        entropy_over_features_after_observing_item_positive = -1 * ((p * np.log(p) + (1 - p) * np.log(1 - p))).sum()
        assert entropy_over_features_after_observing_item_positive > 0, f"Entropy should be positive. Entropy={entropy_over_features_after_observing_item_positive}. Probs={p.round(decimals=3)}"
        self.hypotheses[0].probs = orig_probs 
        # reset learner
        # entropy over features after seeing if negative
#        p, _ = self.hypotheses[0].update(seq, False, verbose=False)

        # reset the judgments to False
        p, results = self.hypotheses[0].update(self.observed_feats+[features], self.observed_judgments+[False], verbose=False, )
        entropy_over_features_after_observing_item_negative = -1 * ((p * np.log(p) + (1 - p) * np.log(1 - p))).sum()
        assert entropy_over_features_after_observing_item_negative > 0, f"Entropy should be positive. Entropy={entropy_over_features_after_observing_item_negative}. Probs={p.round(decimals=3)}"
        self.hypotheses[0].probs = orig_probs 
        # reset learner
#        print("learner after fussing around:", self.hypotheses[0].probs)
        all_equal = (orig_probs == self.hypotheses[0].probs)
        assert all_equal.all()


        # either:
        delta_positive = (entropy_over_features_before_observing_item-entropy_over_features_after_observing_item_positive)
        delta_negative = (entropy_over_features_before_observing_item-entropy_over_features_after_observing_item_negative)
        eig = delta_positive*prob_being_positive + delta_negative*prob_being_negative # reduce my entropy

        # OR

        # the kl-divergence between posteriors before --> observe yes*p(yes) + before --> observe no * p(no)

       # if args.verbose:
       #      print("delta positive:", delta_positive)
       #      print("delta negative:", delta_negative)
       #      print("prob being positive:", prob_being_positive)
        return eig

    def get_kl(self, seq):
        orig_probs = deepcopy(self.hypotheses[0].probs)
#        print("learner before fussing around:", learner_before_fussing_around.probs)

        # prob of the thing being positive or negative
        prob_being_positive_a = np.exp(self.hypotheses[0].logprob(seq, True))
        prob_being_negative_a = np.exp(self.hypotheses[0].logprob(seq, False))
        prob_being_positive = prob_being_positive_a/(prob_being_positive_a+prob_being_negative_a)
        prob_being_negative = 1-prob_being_positive
        assert prob_being_positive + prob_being_negative == 1

        features = self.hypotheses[0]._featurize(seq).nonzero()[0]
        
        p, results = self.hypotheses[0].update(self.observed_feats+[features], self.observed_judgments+[True], verbose=False)
        p_after_true = p.copy()

        # reset learner
        self.hypotheses[0].probs = orig_probs 
        
        p, results = self.hypotheses[0].update(self.observed_feats+[features], self.observed_judgments+[False], verbose=False)
        p_after_false = p.copy()

        # reset learner
        self.hypotheses[0].probs = orig_probs 
        
        all_equal = (orig_probs == self.hypotheses[0].probs)
        assert all_equal.all()

        kl_pos = kl_bern(p_after_true, orig_probs).sum() 
        kl_neg = kl_bern(p_after_false, orig_probs).sum() 

        kl = kl_pos*prob_being_positive + kl_neg*prob_being_negative 
#        print("kl: ", kl)
#        print("kl pos: ", kl_pos)
#        print("kl neg: ", kl_neg)

        q = orig_probs
        p = p_after_true
#        print("p * np.log(p/q)")
#        print((p * np.log(p/q)).round(3))
#        print("(1-p) * np.log((1-p)/(1-q))")
#        print(((1-p) * np.log((1-p)/(1-q))).round(3))
#        print("kl")
#        print(kl_bern(p_after_true, orig_probs).round(3))

        assert kl >= -1e-12, f"The kl divergence is not >= 0: {kl}, kl pos: {kl_pos}, kl_neg: {kl_neg}\norig probs: {orig_probs.round(3)}\np_after_true: {p_after_true.round(3)}\np_after_false: {p_after_false.round(3)}\n{orig_probs==p_after_true}"

        return kl

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

    def all_features(self, return_indices=False):
        feats = []
        hyp = self.hypotheses[0]
        for i, ngram_feat in enumerate(hyp.ngram_features.keys()):
            parts = " :: ".join(hyp.feature_vocab.get_rev(f) for f in ngram_feat)
            if return_indices:
                feats.append((hyp.probs[i].item(), parts, i))
            else:
                feats.append((hyp.probs[i].item(), parts))
        return sorted(feats)

    def get_scores(self, metric, candidates, length_norm):
        if metric == "entropy":
            # TODO: implement multiprocessing; nontrivial bc requires non-local function
            return [self.hypotheses[0].entropy(c, length_norm=length_norm) for c in candidates]
        elif metric == "eig":
            func = self.get_eig
        elif metric == "kl":
            func = self.get_kl
        elif metric == "entropy_pred": 
            func = self.hypotheses[0].entropy_pred
        else:
            raise NotImplementedError
        pool = multiprocessing.Pool()
        scores = pool.map(func, candidates)
        pool.close()
        pool.join()
        return scores

    def get_train_candidate(self, n_candidates, obs_set):
        #print(self.linear_train_dataset)
        while True:
            seq = self.linear_train_dataset[self.index_of_next_item]
            #print("proposing item",seq,"with index",self.index_of_next_item)
            self.index_of_next_item += 1
            #seq = self.dataset.random_example()
            if seq not in obs_set:
                return seq

    def propose(self, n_candidates, forbidden_data, length_norm, verbose=False):
        obs_set_a = set(s for s, _, j in self.observations)
        obs_set = set(s for s in (forbidden_data+list(obs_set_a)))
        
        chosen_strategy = self.strategy_name
        
        # get train
        if np.random.random() < self.propose_train or self.strategy_name == "train":
            return get_train_candidate(self, n_candidates, obs_set)

        candidates = []
        while len(candidates) == 0:
            candidates = [self.dataset.random_seq() for _ in range(n_candidates)]
            candidates = [c for c in candidates if c not in obs_set]
        # TODO: for super strategies, compute based on running averages OR have argument to propose that determines strategy, and if that is not there, default to self.strategy_name

        #print("candidates: ", candidates)
        #import ipdb; ipdb.set_trace()
        if self.strategy_name == "unif" or self.propose_train > 0:
            scores = [0 for c in candidates]
        elif self.strategy_name in ["entropy", "eig", "kl", "entropy_pred"]:
            scores = self.get_scores(self.strategy_name, candidates, length_norm)
        elif self.strategy_name in [
                "eig_train_model", 
                "eig_train_history", 
                "kl_train_model", 
                "kl_train_history"]:
            if self.strategy_name == "eig_train_model":
                metric, is_history = "eig", False
            elif self.strategy_name == "eig_train_history":
                metric, is_history = "eig", True 
            elif self.strategy_name == "kl_train_model":
                metric, is_history = "kl", False
            elif self.strategy_name == "kl_train_history":
                metric, is_history = "kl", True 

            if self.metric_to_track == "kl":
                metrics_by_strategy = self.kls_by_strategy
            elif self.metric_to_track == "entropy_diff":
                metrics_by_strategy = self.entropy_diffs_by_strategy
            else:
                raise ValueError()

            train_mean = np.mean(metrics_by_strategy["train"])
            strategy_mean = np.mean(metrics_by_strategy[metric])

            scores = self.get_scores(metric, candidates, length_norm)
            scored_candidates = list(zip(candidates, scores))
            best_cand, expected_score = max(scored_candidates, key=lambda p: p[1])
           
            # if is history, compare with history (strategy_mean); else expected_score
            comparison_score = strategy_mean if is_history else expected_score
            num_observations = len(self.observations)
            if (not is_history) and (num_observations == 0):
                print("choosing train for the first step...")
                choose_train = True
            elif (is_history and num_observations == 0):
                choose_train = random.sample([True, False], 1)[0]
                print("choosing train for step=0? ", choose_train)
            elif (is_history and num_observations == 1):
                # at second step, choose whatever strategy wasn't already chosen
                choose_train = True if self.chosen_strategies[0] != "train" else False
                print("choosing train for step=1? ", choose_train)
            elif train_mean > comparison_score:
                choose_train = True
            else:
                choose_train = False

            if choose_train:
                train_cand = self.get_train_candidate(n_candidates, forbidden_data)
                chosen_cand, chosen_strategy = train_cand, "train"
            else:
                chosen_cand, chosen_strategy = best_cand, metric

            print(f"chosen strategy: {chosen_strategy}")
            print(f"current train mean: {train_mean} ({metrics_by_strategy['train']})")
            if is_history:
                print(f"current strategy mean: {strategy_mean} ({metrics_by_strategy[metric]})")
            else:
                print(f"expected score: {expected_score}")
            
            # TODO: reorganize so that appending this score happens in the same place for all strategies
            self.chosen_strategies.append(chosen_strategy)
            return chosen_cand
        else:
            raise NotImplementedError(f"strategy {self.strategy_name} not implemented")

        scored_candidates = list(zip(candidates, scores))
        best = max(scored_candidates, key=lambda p: p[1])
        #print(be, self.strategy(best[1]))
        #print(scored_candidates)
        #assert False
        #print(best[1])
        # Print sorted scores
        sorted_scores = sorted([x for x in scored_candidates], key=lambda tup: tup[1], reverse=True)
        print(f"# candidates: {len(sorted_scores)}")
        for c, s in sorted_scores[:5]:
            print(c, s)

        # keep track of which strategies were chosen
        self.chosen_strategies.append(self.strategy_name)
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
class BayesianMaxEntLearner(Learner):
    def __init__(self, dataset, strategy, n_hyps, dim):
        super().__init__(self, dataset, strategy)
