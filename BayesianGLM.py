import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
from scorers import _load_phoneme_features
import itertools as it
import gc
from truncated_normal import tn_kl, tn_entropy


class BayesianScorer:
    def __init__(
        self,
        seed,
        n_features,
        alpha_prior_mu,
        alpha_prior_sigma,
        beta_prior_mu,
        beta_prior_sigma,
        step_size,
        n_updates,
        use_mean,
    ):
        assert alpha_prior_mu > 0
        assert alpha_prior_sigma > 0
        assert beta_prior_mu < 0
        assert beta_prior_sigma > 0
        self.alpha_prior_mu = alpha_prior_mu
        self.alpha_prior_sigma = alpha_prior_sigma
        self.beta_prior_mu = beta_prior_mu
        self.beta_prior_sigma = beta_prior_sigma
        self.step_size = step_size
        self.n_updates = n_updates
        self.rng_key = jax.random.PRNGKey(seed)
        self.use_mean = use_mean
        self.params = {
            "beta_posterior_mu": jnp.full(n_features, self.beta_prior_mu),
            "beta_posterior_sigma": jnp.full(n_features, self.beta_prior_sigma),
            "alpha_posterior_mu": jnp.array(self.alpha_prior_mu),
            "alpha_posterior_sigma": jnp.array(self.alpha_prior_sigma),
        }

    @staticmethod
    def _model(
        data,
        judgments,
        alpha_prior_mu,
        alpha_prior_sigma,
        beta_prior_mu,
        beta_prior_sigma,
    ):
        n_samples, n_features = data.shape

        alpha_prior = dist.TruncatedNormal(
            alpha_prior_mu,
            alpha_prior_sigma,
            low=0,
        )
        beta_prior = dist.TruncatedNormal(
            jnp.full(n_features, beta_prior_mu),
            jnp.full(n_features, beta_prior_sigma),
            high=0,
        )

        alpha = numpyro.sample("alpha", alpha_prior)
        with numpyro.plate("beta_plate", n_features):
            beta = numpyro.sample("beta", beta_prior)

        logits = jnp.matmul(data, beta) + jnp.full(n_samples, alpha)
        likelihood = dist.Bernoulli(logits=logits)
        with numpyro.plate("judgments_plate", n_samples):
            numpyro.sample("judgments", likelihood, obs=judgments)

    @staticmethod
    def _guide(
        data,
        judgments,
        alpha_prior_mu,
        alpha_prior_sigma,
        beta_prior_mu,
        beta_prior_sigma,
    ):
        _, n_features = data.shape

        alpha_posterior_mu = numpyro.param(
            "alpha_posterior_mu",
            alpha_prior_mu,
            constraint=dist.constraints.greater_than(0),
        )
        alpha_posterior_sigma = numpyro.param(
            "alpha_posterior_sigma",
            alpha_prior_sigma,
            constraint=dist.constraints.greater_than(0),
        )
        alpha_posterior = dist.TruncatedNormal(
            alpha_posterior_mu,
            alpha_posterior_sigma,
            low=0,
        )

        beta_posterior_mu = numpyro.param(
            "beta_posterior_mu",
            jnp.full(n_features, beta_prior_mu),
            constraint=dist.constraints.less_than(0),
        )
        beta_posterior_sigma = numpyro.param(
            "beta_posterior_sigma",
            jnp.full(n_features, beta_prior_sigma),
            constraint=dist.constraints.greater_than(0),
        )
        beta_posterior = dist.TruncatedNormal(
            beta_posterior_mu, beta_posterior_sigma, high=0
        )

        numpyro.sample("alpha", alpha_posterior)
        with numpyro.plate("beta_plate", n_features):
            numpyro.sample("beta", beta_posterior)

    @staticmethod
    def compute_posterior(
        data,
        judgments,
        alpha_prior_mu,
        alpha_prior_sigma,
        beta_prior_mu,
        beta_prior_sigma,
        step_size,
        n_updates,
        rng_key,
    ):
        svi = SVI(
            BayesianScorer._model,
            BayesianScorer._guide,
            optim=Adam(step_size=step_size),
            loss=Trace_ELBO(),
        )
        new_params = svi.run(
            rng_key,
            n_updates,
            data,
            judgments,
            alpha_prior_mu,
            alpha_prior_sigma,
            beta_prior_mu,
            beta_prior_sigma,
            progress_bar=False,
            stable_update=True,
        ).params
        return new_params

    def update_posterior(self, data, judgments, update):
        new_params = self.compute_posterior(
            data,
            judgments,
            self.alpha_prior_mu,
            self.alpha_prior_sigma,
            self.beta_prior_mu,
            self.beta_prior_sigma,
            self.step_size,
            self.n_updates,
            self.rng_key,
        )
        if update:
            self.params = new_params
        return new_params

    @staticmethod
    @jax.jit
    def make_posterior(params):
        beta = dist.TruncatedNormal(
            params["beta_posterior_mu"],
            params["beta_posterior_sigma"],
            high=0,
        )
        alpha = dist.TruncatedNormal(
            params["alpha_posterior_mu"],
            params["alpha_posterior_sigma"],
            low=0,
        )
        return beta, alpha

    def get_posterior(self):
        return self.make_posterior(self.params)

    def logits(self, featurized_seq):
        if self.use_mean:
            beta, alpha = self.get_posterior()
            beta_estimate, alpha_estimate = beta.mean, alpha.mean
        else:
            beta_estimate = self.params["beta_posterior_mu"]
            alpha_estimate = self.params["alpha_posterior_mu"]

        logits = jnp.matmul(featurized_seq, beta_estimate) + alpha_estimate
        return logits

    @staticmethod
    def entropy(params, ind=None, length_norm=False):
        if ind is None:
            ind = jnp.arange(len(params["beta_posterior_mu"]))
        beta_entropy = jnp.sum(tn_entropy(
            params["beta_posterior_mu"][ind],
            params["beta_posterior_sigma"][ind],
            b=0,
        )).item()
        alpha_entropy = tn_entropy(
            params["alpha_posterior_mu"],
            params["alpha_posterior_sigma"],
            a=0,
        ).item()
        entropy = beta_entropy + alpha_entropy
        if length_norm:
            entropy /= len(ind) + 1  # (1 for alpha)
        return entropy

    def get_entropy(self, ind=None, length_norm=False):
        return self.entropy(self.params, ind, length_norm)

    @staticmethod
    def info_gain(new_params, old_params):
        old_entropy = BayesianScorer.entropy(old_params)
        new_entropy = BayesianScorer.entropy(new_params)
        info_gain = old_entropy - new_entropy
        return info_gain

    @staticmethod
    def kl_divergence(new_params, old_params):
        beta_kl = jnp.sum(tn_kl(
            new_params["beta_posterior_mu"],
            new_params["beta_posterior_sigma"],
            -jnp.inf,
            0,
            old_params["beta_posterior_mu"],
            old_params["beta_posterior_sigma"],
            -jnp.inf,
            0,
        )).item()
        alpha_kl = tn_kl(
            new_params["alpha_posterior_mu"],
            new_params["alpha_posterior_sigma"],
            0,
            jnp.inf,
            old_params["alpha_posterior_mu"],
            old_params["alpha_posterior_sigma"],
            0,
            jnp.inf,
        ).item()
        kl = beta_kl + alpha_kl
        return kl


class BayesianLearner:
    def __init__(
        self,
        dataset,
        strategy,
        linear_train_dataset,
        index_of_next_item,
        feature_type="atr_harmony",
        phoneme_feature_file=None,
        track_params=False,
        seed=0,
        # scorer hyperparams
        alpha_prior_mu=5.0,
        alpha_prior_sigma=1.0,
        beta_prior_mu=-10.0,
        beta_prior_sigma=20.0,
        step_size=0.01,
        n_updates=2000,
        use_mean=False,
    ):
        assert strategy in {
            "train",
            "unif",
            "entropy",
            "entropy_pred",
            "eig",
            "eig_train_mixed",
            "eig_train_model",
            "eig_train_history",
            "kl",
            "kl_train_mixed",
            "kl_train_model",
            "kl_train_history",
        }

        self.dataset = dataset
        self.strategy = strategy
        self.linear_train_dataset = linear_train_dataset
        self.index_of_next_item = index_of_next_item
        self.track_params = track_params
        self.seed = seed

        self.ORDER = 3
        self._featurized_cache = {}
        self.feature_type = feature_type
        self.phoneme_features, self.feature_vocab = _load_phoneme_features(
            dataset, self.feature_type, phoneme_feature_file=phoneme_feature_file
        )
        self.ngram_features = {}
        for ff in it.product(range(len(self.feature_vocab)), repeat=self.ORDER):
            self.ngram_features[ff] = len(self.ngram_features)
        self.n_features = len(self.ngram_features)

        self.history_based = self.strategy.endswith("history")
        self.model_based = self.strategy.endswith("model")
        self.mixed = self.strategy.endswith("mixed")
        self.metric = self.strategy.split("_")[0]
        self.first_strategies = []
        self.last_proposed = None
        self.n_observed_train = 0
        self.n_observed_metric = 0
        self.train_avg = 0
        self.metric_avg = 0

        if self.history_based:
            self.first_strategies.append(self.metric)
        if self.history_based or self.mixed:
            self.first_strategies.append("train")
        np.random.shuffle(self.first_strategies)

        self.metric_func = None
        if self.metric == "eig":
            self.metric_func = BayesianScorer.info_gain
        if self.metric == "kl":
            self.metric_func = BayesianScorer.kl_divergence

        # scorer hyperparams
        self.alpha_prior_mu = alpha_prior_mu
        self.alpha_prior_sigma = alpha_prior_sigma
        self.beta_prior_mu = beta_prior_mu
        self.beta_prior_sigma = beta_prior_sigma
        self.step_size = step_size
        self.n_updates = n_updates
        self.use_mean = use_mean

    def featurize(self, seq):
        if seq in self._featurized_cache:
            return self._featurized_cache[seq]
        else:
            features = np.zeros(len(self.ngram_features))
            for i in range(len(seq) - self.ORDER + 1):
                features_here = [
                    self.phoneme_features[seq[j]].nonzero()[0]
                    for j in range(i, i + self.ORDER)
                ]
                for ff in it.product(*features_here):
                    features[self.ngram_features[ff]] += 1
#            self._featurized_cache[seq] = features
            return features

    def binary_featurize(self, seq):
        return jnp.array(self.featurize(seq) > 0, float)

    def initialize(self):
        self.observed_judgments = []
        self.observed_features = []
        self.observed_seqs = []
        self.hypothesis = BayesianScorer(
            seed=self.seed,
            n_features=self.n_features,
            alpha_prior_mu=self.alpha_prior_mu,
            alpha_prior_sigma=self.alpha_prior_sigma,
            beta_prior_mu=self.beta_prior_mu,
            beta_prior_sigma=self.beta_prior_sigma,
            step_size=self.step_size,
            n_updates=self.n_updates,
            use_mean=self.use_mean,
        )

        if self.track_params:
            self.observed_feat_idxs = set()
            self.n_seen_feats = []
            self.pct_good_examples = []
            self.alpha_mu = []
            self.alpha_sigma = []
            self.avg_beta_mu = []
            self.avg_beta_sigma = []
            self.avg_seen_beta_mu = []
            self.avg_seen_beta_sigma = []
            self.avg_unseen_beta_mu = []
            self.avg_unseen_beta_sigma = []
            self.proposed_from = []
            self.train_avgs = []
            self.metric_avgs = []
            self.update_param_trackers()

    def observe(self, seq, judgment, update=True):
        featurized = self.binary_featurize(seq)
        self.observed_judgments.append(float(judgment))
        self.observed_features.append(featurized)
        self.observed_seqs.append(seq)

        data = jnp.stack(self.observed_features)
        judgments = jnp.array(self.observed_judgments)
        old_params = self.hypothesis.params
        new_params = self.hypothesis.update_posterior(data, judgments, update)

        if not update:
            self.observed_judgments.pop()
            self.observed_features.pop()
            self.observed_seqs.pop()
        if update and (self.history_based or self.mixed):
            metric_val = self.metric_func(new_params, old_params)
            self.update_history(metric_val)
        if update and self.track_params:
            self.observed_feat_idxs.update(jnp.where(featurized)[0].tolist())
            self.update_param_trackers()
        return new_params

    def update_history(self, metric_val):
        assert self.history_based or self.mixed
        assert self.last_proposed is not None
        if self.last_proposed == "train":
            self.n_observed_train += 1
            self.train_avg += (1 / self.n_observed_train) * (
                metric_val - self.train_avg
            )
        else:
            self.n_observed_metric += 1
            self.metric_avg += (1 / self.n_observed_metric) * (
                metric_val - self.metric_avg
            )

    def propose(self, n_candidates=100, forbidden_seqs=[]):
        exclude = set(self.observed_seqs + forbidden_seqs)

        if len(self.observed_seqs) < len(self.first_strategies):
            if self.first_strategies[len(self.observed_seqs)] == "train":
                self.last_proposed = "train"
                return self.get_train_candidate(exclude)
        elif self.strategy == "train" or (
            self.history_based and self.train_avg >= self.metric_avg
        ):
            self.last_proposed = "train"
            return self.get_train_candidate(exclude)

        # TODO: candidates can come from edits too (not using actually)
        candidates = []
        while len(candidates) == 0:
            candidates = [self.dataset.random_seq() for _ in range(n_candidates)]
            candidates = [c for c in candidates if c not in exclude]

        scores, expected_train = self.score_candidates(candidates)
        scored_candidates = zip(candidates, scores)
        best_cand = max(scored_candidates, key=lambda c: c[1])

        if len(self.observed_seqs) >= len(self.first_strategies):
            if (self.model_based and expected_train >= best_cand[1]) or (
                self.mixed and self.train_avg >= best_cand[1]
            ):
                self.last_proposed = "train"
                return self.get_train_candidate(exclude)

        self.last_proposed = "metric"
        return best_cand[0]

    def get_train_candidate(self, exclude):
        while True:
            seq = self.linear_train_dataset[self.index_of_next_item]
            self.index_of_next_item += 1
            if seq not in exclude:
                return seq

    def score_candidates(self, seqs):
        if self.strategy == "unif":
            return [0] * len(seqs), "NA"
        if self.strategy == "entropy":
            return [self.entropy_param(seq, length_norm=True) for seq in seqs], "NA"
        if self.strategy == "entropy_pred":
            return [self.entropy_pred(seq) for seq in seqs], "NA"
        assert self.metric_func is not None
        return self.get_batch_expected_metric(seqs)

    def get_batch_expected_metric(self, seqs):
        featurized_seqs = [self.binary_featurize(seq) for seq in seqs]
        prob_pos = [self.probs(seq) for seq in seqs]

        @jax.jit
        def compute_posterior(featurized_seq, label):
            return BayesianScorer.compute_posterior(
                jnp.stack(self.observed_features + [featurized_seq]),
                jnp.array(self.observed_judgments + [label]),
                self.hypothesis.alpha_prior_mu,
                self.hypothesis.alpha_prior_sigma,
                self.hypothesis.beta_prior_mu,
                self.hypothesis.beta_prior_sigma,
                self.hypothesis.step_size,
                self.hypothesis.n_updates,
                self.hypothesis.rng_key,
            )
        
        n_jobs = min(multiprocessing.cpu_count() - 1, len(seqs))
        with parallel_backend("loky", n_jobs=n_jobs):
            pos_params = Parallel()(delayed(compute_posterior)(f, 1.0) for f in featurized_seqs)
            gc.collect()
            neg_params = Parallel()(delayed(compute_posterior)(f, 0.0) for f in featurized_seqs)
            gc.collect()
            pos_deltas = Parallel()(delayed(self.metric_func)(p, self.hypothesis.params) for p in pos_params)
            neg_deltas = Parallel()(delayed(self.metric_func)(n, self.hypothesis.params) for n in neg_params)
        
        expected = [
            prob_pos[i] * pos_deltas[i] + (1 - prob_pos[i]) * neg_deltas[i]
            for i in range(len(seqs))
        ]
        expected_train = sum(
            [prob_pos[i] * pos_deltas[i] for i in range(len(seqs))]
        ) / sum(prob_pos)
        return expected, expected_train

    def logits(self, seq):
        featurized_seq = self.binary_featurize(seq)
        logits = self.hypothesis.logits(featurized_seq)
        return logits

    def probs(self, seq):
        logits = self.logits(seq)
        probs = jax.nn.sigmoid(logits)
        return probs

    def logprobs(self, seq):
        probs = self.probs(seq)
        # condition prevents 0 probs / nan logprobs / inf costs
        # works bc logprobs converges to logits when approaching -inf
        logprobs = jnp.log(probs) if probs > 0 else self.logits(seq)
        return logprobs

    def cost(self, seq):
        logprobs = self.logprobs(seq)
        cost = -logprobs
        return cost

    def entropy_pred(self, seq):
        p = self.probs(seq)
        entropy = -p * jnp.log(p) - ((1 - p) * jnp.log(1 - p))
        return entropy

    def entropy_param(self, seq, length_norm=False):
        ind = jnp.where(self.binary_featurize(seq))[0]
        return self.hypothesis.get_entropy(ind, length_norm)

    def update_param_trackers(self):
        p = self.hypothesis.params
        seen_feats = np.array(list(self.observed_feat_idxs))
        unseen_feats = np.array(
            [i for i in range(self.n_features) if i not in self.observed_feat_idxs]
        )

        self.n_seen_feats.append(len(seen_feats))
        self.pct_good_examples.append(
            sum(self.observed_judgments) / len(self.observed_judgments)
            if len(self.observed_judgments)
            else 0
        )
        self.alpha_mu.append(p["alpha_posterior_mu"])
        self.alpha_sigma.append(p["alpha_posterior_sigma"])
        self.avg_beta_mu.append(p["beta_posterior_mu"].mean())
        self.avg_beta_sigma.append(p["beta_posterior_sigma"].mean())
        self.avg_seen_beta_mu.append(
            p["beta_posterior_mu"][seen_feats].mean() if len(seen_feats) else np.nan
        )
        self.avg_seen_beta_sigma.append(
            p["beta_posterior_sigma"][seen_feats].mean() if len(seen_feats) else np.nan
        )
        self.avg_unseen_beta_mu.append(
            p["beta_posterior_mu"][unseen_feats].mean() if len(unseen_feats) else np.nan
        )
        self.avg_unseen_beta_sigma.append(
            p["beta_posterior_sigma"][unseen_feats].mean()
            if len(unseen_feats)
            else np.nan
        )
        self.proposed_from.append(self.last_proposed)
        self.train_avgs.append(self.train_avg)
        self.metric_avgs.append(self.metric_avg)

    def get_param_trackers(self):
        return {
            "n_seen_feats": self.n_seen_feats,
            "pct_good_examples": self.pct_good_examples,
            "alpha_mu": self.alpha_mu,
            "alpha_sigma": self.alpha_sigma,
            "avg_beta_mu": self.avg_beta_mu,
            "avg_beta_sigma": self.avg_beta_sigma,
            "avg_seen_beta_mu": self.avg_seen_beta_mu,
            "avg_seen_beta_sigma": self.avg_seen_beta_sigma,
            "avg_unseen_beta_mu": self.avg_unseen_beta_mu,
            "avg_unseen_beta_sigma": self.avg_unseen_beta_sigma,
            "proposed_from": self.proposed_from,
            "train_avgs": self.train_avgs,
            "metric_avgs": self.metric_avgs,
        }
