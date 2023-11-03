import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
import multiprocessing
from scorers import _load_phoneme_features
import itertools as it


class BayesianScorer:
    def __init__(
        self,
        n_features = 512,
        alpha_prior_mu = 3.0,
        # to "constrain" alpha optimization (might be worth making alpha fixed)
        alpha_prior_sigma = 0.0001,
        beta_prior_mu = -1.0,
        beta_prior_sigma = 10.0,
        step_size = 0.01,
        n_updates = 1000,
        seed = 0,
    ) -> None:
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
        self.params = {
            "beta_posterior_mu": jnp.full(n_features, self.beta_prior_mu),
            "beta_posterior_sigma": jnp.full(n_features, self.beta_prior_sigma),
            "alpha_posterior_mu": self.alpha_prior_mu,
            "alpha_posterior_sigma": self.alpha_prior_sigma,
        }

    @staticmethod
    def _model(
        data,
        judgments,
        alpha_prior_mu,
        alpha_prior_sigma,
        beta_prior_mu,
        beta_prior_sigma
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
            high=0
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
        beta_prior_sigma
    ):
        _, n_features = data.shape

        alpha_posterior_mu = numpyro.param(
            "alpha_posterior_mu",
            alpha_prior_mu,
            constraint=dist.constraints.greater_than(0)
        )
        alpha_posterior_sigma = numpyro.param(
            "alpha_posterior_sigma",
            alpha_prior_sigma,
            constraint=dist.constraints.greater_than(0)
        )
        alpha_posterior = dist.TruncatedNormal(
            alpha_posterior_mu,
            alpha_posterior_sigma,
            low=0,
        )

        beta_posterior_mu = numpyro.param(
            "beta_posterior_mu",
            jnp.full(n_features, beta_prior_mu),
            constraint=dist.constraints.less_than(0)
        )
        beta_posterior_sigma = numpyro.param(
            "beta_posterior_sigma",
            jnp.full(n_features, beta_prior_sigma),
            constraint=dist.constraints.greater_than(0)
        )
        beta_posterior = dist.TruncatedNormal(
             beta_posterior_mu,
             beta_posterior_sigma,
             high=0
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
        rng_key
    ):
        svi = SVI(
            BayesianScorer._model,
            BayesianScorer._guide,
            optim=Adam(step_size=step_size),
            loss=Trace_ELBO()
        )
        # has some weird startup time but then is fast (almost) regardless of n_updates
        new_params = svi.run(
            rng_key,
            n_updates,
            data,
            judgments,
            alpha_prior_mu,
            alpha_prior_sigma,
            beta_prior_mu,
            beta_prior_sigma,
            progress_bar=False
        ).params
        return new_params
    
    def update_posterior(
        self,
        data,
        judgments,
        update
    ):
        new_params = self.compute_posterior(
            data,
            judgments,
            self.alpha_prior_mu,
            self.alpha_prior_sigma,
            self.beta_prior_mu,
            self.beta_prior_sigma,
            self.step_size,
            self.n_updates,
            self.rng_key            
        )
        if update:
            self.params = new_params
        return new_params
    
    @staticmethod
    def make_posterior(params):
        beta = dist.TruncatedNormal(
            params["beta_posterior_mu"],
            params["beta_posterior_sigma"],
            high=0
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
        logits = jnp.matmul(
            featurized_seq,
            # QUESTION: should these be mu (effectively mode) or true mean of truncated dist?
            self.params["beta_posterior_mu"]
        ) + self.params["alpha_posterior_mu"]
        return logits
    
    @staticmethod
    def truncated_normal_entropy(mu, sigma, low=-jnp.inf, high=jnp.inf):
    # https://en.wikipedia.org/wiki/Truncated_normal_distribution#:~:text=The%20truncated%20normal%20is%20one,support%20form%20an%20exponential%20family.
        phi = lambda x: (1 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * jnp.square(x))
        big_phi = lambda x: 0.5 * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))
        mult = lambda x, y: jnp.nan_to_num(
            x * y, nan=0, posinf=jnp.inf, neginf=-jnp.inf
        )  # makes 0 * inf = 0 instead of nan

        alpha = (low - mu) / sigma
        beta = (high - mu) / sigma
        Z = big_phi(beta) - big_phi(alpha)
        entropy = jnp.log(jnp.sqrt(2 * jnp.pi * jnp.e) * sigma * Z) + (
            mult(alpha, phi(alpha)) - mult(beta, phi(beta))
        ) / (2 * Z)
        return entropy

    @staticmethod
    def entropy(params):
        beta_entropy = jnp.sum(BayesianScorer.truncated_normal_entropy(
            params["beta_posterior_mu"],
            params["beta_posterior_sigma"],
            high=0
        )).item() 
        alpha_entropy = BayesianScorer.truncated_normal_entropy(
            params["alpha_posterior_mu"],
            params["alpha_posterior_sigma"],
            low=0
        ).item()
        entropy = beta_entropy + alpha_entropy
        return entropy
    
    @staticmethod
    def info_gain(new_params, old_params):
        old_entropy = BayesianScorer.entropy(old_params)
        new_entropy = BayesianScorer.entropy(new_params)
        info_gain = old_entropy - new_entropy
        return info_gain
    
    @staticmethod
    def truncated_normal_kl(mu1, sigma1, low1, high1, mu2, sigma2, low2, high2):
        # https://doi.org/10.3390%2Fe24030421 (eq. 111)
        phi = lambda x: (1 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * jnp.square(x))
        big_phi = lambda x: 0.5 * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))
        big_phi_m_s = lambda x, m, s: 0.5 * (
            1 + jax.scipy.special.erf((x - m) / (jnp.sqrt(2) * s))
        )
        mult = lambda x, y: jnp.nan_to_num(x * y, nan=0, posinf=jnp.inf, neginf=-jnp.inf)

        alpha_f = lambda m, s, a, b: (a - m) / s
        beta_f = lambda m, s, a, b: (b - m) / s

        mu_f = lambda m, s, a, b: m - s * (
            (phi(beta_f(m, s, a, b)) - phi(alpha_f(m, s, a, b)))
            / (big_phi(beta_f(m, s, a, b)) - big_phi(alpha_f(m, s, a, b)))
        )
        var_f = lambda m, s, a, b: jnp.square(s) * (
            1
            - (
                (
                    mult(beta_f(m, s, a, b), phi(beta_f(m, s, a, b)))
                    - mult(alpha_f(m, s, a, b), phi(alpha_f(m, s, a, b)))
                )
                / (big_phi(beta_f(m, s, a, b)) - big_phi(alpha_f(m, s, a, b)))
            )
            - jnp.square(
                (phi(beta_f(m, s, a, b)) - phi(alpha_f(m, s, a, b)))
                / (big_phi(beta_f(m, s, a, b)) - big_phi(alpha_f(m, s, a, b)))
            )
        )
        eta_1 = lambda m, s, a, b: mu_f(m, s, a, b)
        eta_2 = lambda m, s, a, b: var_f(m, s, a, b) + jnp.square(mu_f(m, s, a, b))
        z_a_b = (
            lambda m, s, a, b: jnp.sqrt(2 * jnp.pi)
            * s
            * (big_phi_m_s(b, m, s) - big_phi_m_s(a, m, s))
        )
        kl = (
            (mu2 / 2 * jnp.square(sigma2))
            - (mu1 / 2 * jnp.square(sigma1))
            + jnp.log(z_a_b(mu2, sigma2, low2, high2) / z_a_b(mu1, sigma1, low1, high1))
            - (mu2 / jnp.square(sigma2) - mu1 / jnp.square(sigma1))
            * eta_1(mu1, sigma1, low1, high1)
            - (1 / (2 * jnp.square(sigma1)) - 1 / (2 * jnp.square(sigma2)))
            * eta_2(mu1, sigma1, low1, high1)
        )
        return kl
    
    @staticmethod
    def kl_divergence(new_params, old_params):
        beta_kl = jnp.sum(BayesianScorer.truncated_normal_kl(
            new_params["beta_posterior_mu"],
            new_params["beta_posterior_sigma"],
            -jnp.inf,
            0,
            old_params["beta_posterior_mu"],
            old_params["beta_posterior_sigma"],
            -jnp.inf,
            0,
        )).item()
        alpha_kl = BayesianScorer.truncated_normal_kl(
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
        feature_type = "atr_harmony",
        phoneme_feature_file = None,
        track_params = False,
        seed = 0
    ):
        self.dataset = dataset
        self.strategy = strategy
        self.linear_train_dataset = linear_train_dataset
        self.index_of_next_item = index_of_next_item
        self.track_params = track_params
        self.seed = seed
        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)

        self.ORDER = 3
        self._featurized_cache = {}
        self.feature_type = feature_type
        self.phoneme_features, self.feature_vocab = _load_phoneme_features(
            dataset,
            self.feature_type,
            phoneme_feature_file = phoneme_feature_file
        )
        self.ngram_features = {}
        for ff in it.product(range(len(self.feature_vocab)), repeat=self.ORDER):
            self.ngram_features[ff] = len(self.ngram_features)
        self.n_features = len(self.ngram_features)
      
    def featurize(self, seq):
        if seq in self._featurized_cache:
            return self._featurized_cache[seq]
        else:
            features = np.zeros(len(self.ngram_features))
            for i in range(len(seq) - self.ORDER + 1):
                features_here = [self.phoneme_features[seq[j]].nonzero()[0] for j in range(i, i+self.ORDER)]
                for ff in it.product(*features_here):
                    features[self.ngram_features[ff]] += 1
            self._featurized_cache[seq] = features
            return features
        
    def binary_featurize(self, seq):
        return jnp.array(self.featurize(seq) > 0, float)
        
    def initialize(self):
        self.observed_judgments = []
        self.observed_features = []
        self.observed_seqs = []
        self.hypothesis = BayesianScorer(
            n_features=self.n_features,
            seed=self.seed
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

    def observe(
        self,
        seq,
        judgment,
        update=True
    ):
        featurized = self.binary_featurize(seq)
        self.observed_judgments.append(float(judgment))
        self.observed_features.append(featurized)
        self.observed_seqs.append(seq)

        data = jnp.stack(self.observed_features)
        judgments = jnp.array(self.observed_judgments)
        new_params = self.hypothesis.update_posterior(data, judgments, update)

        if not update:
            self.observed_judgments.pop()
            self.observed_features.pop()
            self.observed_seqs.pop()
        if update and self.track_params:
            self.observed_feat_idxs.update(seq)
            self.update_param_trackers()
        return new_params
    
    def propose(self, n_candidates=100, forbidden_seqs=[]):
        exclude = set(self.observed_seqs + forbidden_seqs)
        # TODO: add other strategies (to score_candidates too)
        if self.strategy == "train":
            return self.get_train_candidate(exclude)

        # TODO: candidates can come from edits too (not using actually)
        candidates = []
        while len(candidates) == 0:
            candidates = [self.dataset.random_seq() for _ in range(n_candidates)]
            candidates = [c for c in candidates if c not in exclude]

        scored_candidates = zip(candidates, self.score_candidates(candidates))
        return max(scored_candidates, key=lambda c: c[1])[0]
    
    def get_train_candidate(self, exclude):
        while True:
            seq = self.linear_train_dataset[self.index_of_next_item]
            self.index_of_next_item += 1
            if seq not in exclude:
                return seq
            
    def score_candidates(self, seqs):
        if self.strategy == "unif":
            return [0]*len(seqs)
        if self.strategy == "eig":
            return self.get_batch_expected_metric(seqs, BayesianScorer.info_gain)
        if self.strategy == "kl":
            return self.get_batch_expected_metric(seqs, BayesianScorer.kl_divergence)
        raise ValueError(f"No score method implemented for strategy '{self.strategy}'")
    
    def get_batch_expected_metric(self, seqs, metric):
        featurized_seqs = [self.binary_featurize(seq) for seq in seqs]

        make_input = lambda f, l: (
            jnp.stack(self.observed_features + [f]),
            jnp.array(self.observed_judgments + [l]),
            self.hypothesis.alpha_prior_mu,
            self.hypothesis.alpha_prior_sigma,
            self.hypothesis.beta_prior_mu,
            self.hypothesis.beta_prior_sigma,
            self.hypothesis.step_size,
            self.hypothesis.n_updates,
            self.hypothesis.rng_key
        )

        prob_pos = [self.probs(seq) for seq in seqs]
        pos_inputs = [make_input(f, 1.0) for f in featurized_seqs]
        neg_inputs = [make_input(f, 0.0) for f in featurized_seqs]

        current_params = [self.hypothesis.params for _ in range(len(seqs))]
        # TODO: investigate using chunksize for increasing speed
        pos_params = self.pool.starmap(BayesianScorer.compute_posterior, pos_inputs)
        neg_params = self.pool.starmap(BayesianScorer.compute_posterior, neg_inputs)
        pos_deltas = self.pool.starmap(metric, zip(pos_params, current_params))
        neg_deltas = self.pool.starmap(metric, zip(neg_params, current_params))

        expected = [prob_pos[i] * pos_deltas[i] + (1-prob_pos[i]) * neg_deltas[i] for i in range(len(seqs))]
        return expected

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
        logprobs = jnp.log(probs)
        return logprobs
    
    def cost(self, seq):
        logprobs = self.logprobs(seq)
        cost = -logprobs
        return cost
    
    def update_param_trackers(self):
        p = self.hypothesis.params
        seen_feats = np.array(list(self.observed_feat_idxs))
        unseen_feats = np.array([i for i in range(self.n_features) if i not in self.observed_feat_idxs])

        self.n_seen_feats.append(len(seen_feats))
        self.pct_good_examples.append(sum(self.observed_judgments)/len(self.observed_judgments))
        self.alpha_mu.append(p["alpha_posterior_mu"])
        self.alpha_sigma.append(p["alpha_posterior_sigma"])
        self.avg_beta_mu.append(p["beta_posterior_mu"].mean())
        self.avg_beta_sigma.append(p["beta_posterior_sigma"].mean())
        self.avg_seen_beta_mu.append(p["beta_posterior_mu"][seen_feats].mean())
        self.avg_seen_beta_sigma.append(p["beta_posterior_sigma"][seen_feats].mean())
        self.avg_unseen_beta_mu.append(p["beta_posterior_mu"][unseen_feats].mean())
        self.avg_unseen_beta_sigma.append(p["beta_posterior_sigma"][unseen_feats].mean())

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
            "avg_unseen_beta_sigma": self.avg_unseen_beta_sigma
        }