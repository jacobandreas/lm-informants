import itertools as it
import numpy as np
import re
from tqdm import tqdm

class HWInformant:
    def __init__(self):
        self.phoneme_feature_index, self.phoneme_index, self.phoneme_features = self._load_phoneme_features()
        self.BOUNDARY = self.phoneme_index["$"]

        #self.transition_feature_index, self.transition_weights = self._load_transition_features()
        self.transition_feature_index, self.transition_weights = self._load_bigram_features()
        #self.transition_feature_index, self.transition_weights = self._load_unigram_features()

        self.n_features = len(self.transition_feature_index)
        self.n_phonemes = len(self.phoneme_index)
        self.rev_phoneme_index = {v: k for k, v in self.phoneme_index.items()}

    def _load_phoneme_features(self):
        feature_index = {}
        phoneme_index = {}
        phoneme_features = {}

        phoneme_index["$"] = 0

        with open("data/hw/phoneme_features.txt") as reader:
            header = next(reader)
            feat_names = header.strip().split("\t")
            feat_names.append("word_boundary")
            for feat_name in feat_names:
                feature_index["+" + feat_name] = len(feature_index)
                feature_index["-" + feat_name] = len(feature_index)
            for phoneme_feats in reader:
                phoneme_feats = phoneme_feats.strip().split("\t")

                phoneme_name = phoneme_feats[0]
                if phoneme_name not in ("B", "T", "M", "L", "IY", "AH", "UW"):
                    continue

                phoneme_index[phoneme_name] = len(phoneme_index)
                phoneme_vec = np.zeros(len(feature_index))
                for i, feat_val in enumerate(phoneme_feats[1:]):
                    if feat_val == "0":
                        continue
                    feat_name = feat_names[i] 
                    feat = feat_val + feat_name
                    phoneme_vec[feature_index[feat]] = 1
                phoneme_vec[feature_index["-word_boundary"]] = 1
                phoneme_features[phoneme_index[phoneme_name]] = phoneme_vec

        boundary_vec = np.zeros(len(feature_index))
        boundary_vec[feature_index["+word_boundary"]] = 1
        phoneme_features[phoneme_index["$"]] = boundary_vec

        return feature_index, phoneme_index, phoneme_features

    def _load_transition_features(self):
        transition_feature_index = {}
        feature_weights = {}
        for feat in it.product(self.phoneme_feature_index.values(), repeat=2):
            feat = ((feat[0],), (feat[1],))
            transition_feature_index[feat] = len(transition_feature_index)
        for feat in it.product(self.phoneme_feature_index.values(), repeat=4):
            feat = ((feat[0], feat[1]), (feat[2], feat[3]))
            transition_feature_index[feat] = len(transition_feature_index)
        with open("data/hw/feature_weights.txt") as reader:
            for line in reader:
                line = line.split("\t")
                features = re.findall(r"\[([^\[\]]+)\]", line[0])
                feature_template = []
                for feature in features:
                    feature_template_part = []
                    parts = feature.split(",")
                    for part in parts:
                        feature_template_part.append(self.phoneme_feature_index[part])
                    feature_template.append(tuple(feature_template_part))
                weight = float(line[-1])
                feature_template = tuple(feature_template)
                #feature_id = len(transition_feature_index)
                #transition_feature_index[feature_template] = feature_id
                if feature_template not in transition_feature_index:
                    continue
                feature_id = transition_feature_index[feature_template]
                feature_weights[feature_id] = weight
        return transition_feature_index, feature_weights


    def _load_bigram_features(self):
        transition_feature_index = {}
        for feat in it.product(self.phoneme_index.values(), repeat=2):
            transition_feature_index[feat] = len(transition_feature_index)
        feature_weights = np.ones(len(transition_feature_index))
        n = 0
        with open("data/hw/words.txt") as reader:
            for line in reader:
                line = line.strip().split()
                if not all(p in self.phoneme_index for p in line):
                    continue
                n += 1
                print(line, n)
                phonemes = self.encode(line)
                for i in range(len(phonemes)-1):
                    feat = (phonemes[i], phonemes[i+1])
                    feature_weights[transition_feature_index[feat]] = 0
        return transition_feature_index, feature_weights


    def _load_unigram_features(self):
        transition_feature_index = {}
        for feat in self.phoneme_index.values():
            if feat == 0:
                continue
            transition_feature_index[feat] = len(transition_feature_index)
        feature_weights = np.ones(len(transition_feature_index))
        with open("data/hw/fake_words.txt") as reader:
            for line in reader:
                phonemes = self.encode(line.strip().split())
                for c in phonemes[1:-1]:
                    feature_weights[transition_feature_index[c]] = 0
        return transition_feature_index, feature_weights



    def encode(self, word):
        return (
            [self.BOUNDARY]
            + [self.phoneme_index[p] for p in word]
            + [self.BOUNDARY]
        )


    def decode(self, word):
        assert word[0] == self.BOUNDARY
        assert word[-1] == self.BOUNDARY
        return " ".join(self.rev_phoneme_index[i] for i in word[1:-1])


    def score(self, word):
        return np.dot(self.featurize(word), self.transition_weights)
        #total = 0
        #features = self.featurize(word)
        #for f, w in self.transition_weights.items():
        #    total += features[f] * w
        #return total


    def judge(self, word):
        return self.score(word) == 0


    #def featurize(self, word):
    #    word_features = np.zeros(len(self.transition_feature_index))
    #    for feature, feature_id in self.transition_feature_index.items():
    #        for i in range(len(word) - len(feature)):
    #            success_here = True
    #            for j in range(len(feature)):
    #                phoneme_features = self.phoneme_features[word[i+j]]
    #                success_here &= all(phoneme_features[f] for f in feature[j])
    #            word_features[feature_id] += success_here
    #    return word_features

    def featurize(self, word):
        features = np.zeros(len(self.transition_feature_index))
        for i in range(len(word)-1):
            feat = (word[i], word[i+1])
            features[self.transition_feature_index[feat]] += 1
        return features

    #def featurize(self, word):
    #    features = np.zeros(len(self.transition_feature_index))
    #    for c in word[1:-1]:
    #        features[self.transition_feature_index[c]] += 1
    #    return features


class HWLearner:
    def __init__(self, informant, random_seed=0):
        self.n_hypotheses = 10
        self.random = np.random.RandomState(random_seed)
        self.hypotheses = [
            (self.random.random(informant.n_features) - 0.5)
            for _ in range(self.n_hypotheses)
        ]
        self.data = []
        self.informant = informant


    def _random_word(self):
        length = self.random.randint(1, 5)
        return (
            [self.informant.BOUNDARY]
            + [self.random.randint(1, self.informant.n_phonemes) for _ in range(length)]
            + [self.informant.BOUNDARY]
        )


    def generate_candidate(self, rule, forbidden=set()):
        if rule == "uniform":
            while True:
                word = tuple(self._random_word())
                if word not in forbidden:
                    return word, None, 0

        candidates = []
        while len(candidates) == 0:
            candidates = [self._random_word() for _ in range(50)]
            candidates = list(set([tuple(c) for c in candidates]))
            candidates = [c for c in candidates if c not in forbidden]

        def diff(ss):
            return max(ss) - min(ss)

        if rule == "diff":
            elig_fn = diff
        elif rule == "var":
            elig_fn = np.var
        elif rule == "max":
            elig_fn = max
        elif rule == "min":
            elig_fn = min
        else:
            assert False

        scores = [1/(1+np.exp(-self.score(c))) for c in candidates]
        eligs = [elig_fn(s) for s in scores]
        ranked_candidates = sorted(
            zip(candidates, scores, eligs), 
            key=lambda p: p[2]
        )
        #ranked_candidates = ranked_candidates[:3] + ranked_candidates[-3:]
        #for c, s, e in ranked_candidates:
        #    print(self.informant.decode(c), s, e)
        #print("---")
        return ranked_candidates[-1]

    def score(self, word):
        f = self.informant.featurize(word)
        scores = np.array([np.dot(h, f) for h in self.hypotheses])
        return scores


    def update(self, word, judgment):
        self.data.append((word, judgment))
        self._run_mcmc()
        #for w, j in self.data:
        #    print(self.informant.decode(w), j, self.score(w))

        #print(np.mean([np.linalg.norm(w - self.informant.transition_weights) for w in self.hypotheses]))


    def evaluate(self):
        eval_random = np.random.RandomState(0)

        good = []
        bad = []
        real = []
        shuf = []
        for datum, judgment in self.data:
            score = self.score(datum)
            (good if judgment else bad).append(score)

        with open("data/hw/words.txt") as reader:
            for line in reader:
                word = line.strip().split()
                if not all(w in self.informant.phoneme_index for w in word):
                    continue
                score = self.score(self.informant.encode(word))
                real.append(score)
                eval_random.shuffle(word)
                score_shuf = self.score(self.informant.encode(word))
                shuf.append(score_shuf)

        #with open("data/hw/fake_words.txt") as reader:
        #    for line in reader:
        #        word = line.strip().split()
        #        score = self.score(self.informant.encode(word))
        #        real.append(score)

        #with open("data/hw/fake_words_bad.txt") as reader:
        #    for line in reader:
        #        word = line.strip().split()
        #        score_shuf = self.score(self.informant.encode(word))
        #        shuf.append(score_shuf)

        return {
            "good": np.mean(good),
            "bad": np.mean(bad),
            "real": np.mean(real),
            "shuf": np.mean(shuf),
        }


    def _run_mcmc(self):
        new_hypotheses = []
        for hyp in (self.hypotheses):
            changed = False
            #if self.random.random() < 0.05:
            #    print("RESET")
            #    hyp = self.random.random(len(hyp)) - 0.5
            orig_ll = curr_ll = self._log_likelihood(hyp)
            i = 0
            while i < 100:
                proposal = hyp + (self.random.random(len(hyp)) - 0.5)
                proposal = np.maximum(proposal, -2)
                proposal = np.minimum(proposal, 2)

                proposed_ll = self._log_likelihood(proposal)
                accept_crit = np.exp(10 * (proposed_ll - curr_ll))
                rand = self.random.random()
                if rand < accept_crit:
                    hyp = proposal
                    curr_ll = proposed_ll
                    changed = True
                i += 1
            new_hypotheses.append(hyp)
            if not changed:
                print("warning: no update")
            if curr_ll < orig_ll:
                print("warning: got worse", orig_ll, curr_ll)
            #else:
            #    print(orig_ll, "->", curr_ll)

        self.hypotheses = new_hypotheses

    def _log_likelihood(self, hyp):
        ll = - 0.001 * np.linalg.norm(hyp)
        if len(self.data) == 0:
            return ll
        score = 0
        ok = 1
        for (word, judgment) in self.data:
            f = self.informant.featurize(word)
            sign = 1 if judgment else -1
            pred = np.dot(f, hyp) < 1
            ok &= (pred == judgment)
            ll += -np.log(1+np.exp(-sign * np.dot(f, hyp)))
        #return np.log(ok)
        return ll + score / len(self.data) 
