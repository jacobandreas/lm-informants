import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from scorers import _load_phoneme_features
import itertools as it
import matplotlib.pyplot as plt
import multiprocessing


default_hyperparams = {
  "alpha_prior_mean": 10, # pos prior
  "alpha_prior_std": 20,
  "beta_prior_mean": -10, # neg prior
  "beta_prior_std": 20,
  "learning_rate": 0.01,
  "max_epochs": 1000,
  "min_epochs": 200,
  "epsilon": 0.05,
}


class BayesianScorer:
    def __init__(self,
                 n_features,
                 hyperparams=default_hyperparams,
                 state = None,
        ):
        self.hyperparams = hyperparams
        self.n_features = n_features
        self.param_init = lambda n, val=None: lambda: torch.full((n,), val) if val else torch.randn(n)

        if state:
            self.set_state(state)
        else:
            self._guide() # initializes posterior parameters
        

    def _model(self, data, judgments=None):
        n_samples, n_features = data.shape
        assert n_features == self.n_features

        alpha_prior = dist.Normal(torch.tensor([self.hyperparams["alpha_prior_mean"]]), 
                                   torch.tensor([self.hyperparams["alpha_prior_std"]]))
        beta_prior = dist.Normal(torch.full((n_features,), self.hyperparams["beta_prior_mean"]), 
                                   torch.full((n_features,), self.hyperparams["beta_prior_std"]))

        alpha = pyro.sample("alpha", alpha_prior)
        with pyro.plate("beta_plate", n_features):
            beta = pyro.sample("beta", beta_prior)

        logits = torch.matmul(data, beta) + torch.full((n_samples,), alpha.item())
        likelihood = dist.Bernoulli(logits=logits)

        with pyro.plate("judgments_plate", n_samples):
            pyro.sample("judgments", likelihood, obs=judgments)
    

    def _guide(self, data=None, judgments=None):
        beta_posterior_mean = pyro.param("beta_posterior_mean", self.param_init(self.n_features, -10), constraint=dist.constraints.less_than(0.0)) # temp neg constraint
        beta_posterior_std = pyro.param("beta_posterior_std", self.param_init(self.n_features, 1), constraint=dist.constraints.positive)
        beta_posterior = dist.Normal(beta_posterior_mean, beta_posterior_std)

        alpha_posterior_mean = pyro.param("alpha_posterior_mean", self.param_init(1, 10), constraint=dist.constraints.positive) # temp pos constraint
        alpha_posterior_std = pyro.param("alpha_posterior_std", self.param_init(1, 1), constraint=dist.constraints.positive)
        alpha_posterior = dist.Normal(alpha_posterior_mean, alpha_posterior_std)

        pyro.sample("alpha", alpha_posterior)
        with pyro.plate("beta_plate", self.n_features):
            pyro.sample("beta", beta_posterior)


    def set_state(self, state):
        pyro.get_param_store().set_state(state)


    def get_state(self):
        state = pyro.get_param_store().get_state()
        state['params'] = {k:v.detach() for k,v in state['params'].items()}
        return state


    def get_params(self):
        return {k:v.detach() for k,v in pyro.get_param_store().items()}
    

    def clear_params(self):
        pyro.clear_param_store()
    

    def compute_params(self, data, judgments, update=True):
        old_state = self.get_state()
        self.clear_params()

        optimizer = Adam({'lr': self.hyperparams["learning_rate"]})
        svi = SVI(self._model, self._guide, optimizer, loss=Trace_ELBO())

        epochs = 0
        avg_loss = 0
        avg_loss_slope = 0 
        min_loss = torch.inf
        while epochs < self.hyperparams["max_epochs"]:
            loss = svi.step(data, judgments)
            if loss < min_loss:
                min_loss = loss
                new_state = self.get_state()
                new_params = self.get_params()

            # TODO: finalize stop criteria (likely replace this, although works pretty well)
            epochs += 1
            if epochs == 1:
                avg_loss = loss
            else:
                prev_avg_loss = avg_loss
                gamma_1 = 2/51
                avg_loss += (loss - avg_loss)*gamma_1
                if epochs == 2:
                    avg_loss_slope = avg_loss - prev_avg_loss
                else:
                    gamma_2 = 2/101
                    avg_loss_slope += (avg_loss - prev_avg_loss - avg_loss_slope)*gamma_2

            if epochs > self.hyperparams["min_epochs"] and abs(avg_loss_slope) < self.hyperparams["epsilon"]:
                break
        
        if not update:
            self.set_state(old_state)
        else:
            self.set_state(new_state)
        return new_params, new_state
    

    def cost(self, featurized_seq):
        return -self.logprobs(featurized_seq)
    

    def logprobs(self, featurized_seq):
        prob = self.probs(featurized_seq)
        return torch.log(prob)
    
    
    def probs(self, featurized_seq):
        logit = self.logits(featurized_seq)
        prob = torch.sigmoid(logit)
        return prob
    
    def logits(self, featurized_seq):
        logit = torch.dot(featurized_seq, pyro.param("beta_posterior_mean")) + pyro.param("alpha_posterior_mean")
        return logit.detach()

class BayesianLearner:
    def __init__(self, dataset, strategy, linear_train_dataset, index_of_next_item, feature_type="atr_harmony", phoneme_feature_file=None):
        self.dataset = dataset
        self.strategy = strategy
        self.linear_train_dataset = linear_train_dataset
        self.index_of_next_item = index_of_next_item
        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        
        self.ORDER = 3
        self._featurized_cache = {}
        self.feature_type = feature_type

        self.phoneme_features, self.feature_vocab = _load_phoneme_features(dataset, self.feature_type, phoneme_feature_file=phoneme_feature_file)
        self.ngram_features = {}
        for ff in it.product(range(len(self.feature_vocab)), repeat=self.ORDER):
            self.ngram_features[ff] = len(self.ngram_features)
    
        self.n_features = len(self.ngram_features)


    # moved featurization to learner (made more sense to me and made scorer more lightweight)
    def featurize(self, seq):
        if seq in self._featurized_cache:
            return self._featurized_cache[seq]
        else:
            features = torch.zeros(len(self.ngram_features))
            for i in range(len(seq) - self.ORDER + 1):
                features_here = [self.phoneme_features[seq[j]].nonzero()[0] for j in range(i, i+self.ORDER)]
                for ff in it.product(*features_here):
                    features[self.ngram_features[ff]] += 1 # not currently binary, should it be?
            self._featurized_cache[seq] = features
            return features
    

    def initialize(self):
        self.observed_judgments = []
        self.observed_features = []
        self.observed_seqs = []
        self.hypothesis = self.initialize_hyp() # normal implementation is "hypotheses" but i've only ever seen 1


    def initialize_hyp(self):
        return BayesianScorer(self.n_features)
    
    
    def observe(self, seq, judgment, update=True):
        featurized = self.featurize(seq)
        self.observed_judgments.append(float(judgment))
        self.observed_features.append(featurized)
        self.observed_seqs.append(seq)

        data = torch.stack(self.observed_features)
        judgments = torch.tensor(self.observed_judgments)
        new_params = self.hypothesis.compute_params(data, judgments, update)
        if not update:
            # do we keep as observed if not chosen?
            self.observed_judgments.pop()
            self.observed_features.pop()
            self.observed_seqs.pop()
        return new_params
    

    def get_train_candidate(self, forbidden_seqs):
        exclude = set(self.observed_seqs + forbidden_seqs)
        while True:
            seq = self.linear_train_dataset[self.index_of_next_item]
            self.index_of_next_item += 1
            if seq not in exclude:
                return seq


    def score_candidates(self, seqs):
        # TODO: score seq according to metric, which might call propose without update
        if self.strategy == "unif":
            return [0]*len(seqs)
        if self.strategy == "eig":
            return self.get_batch_expected_metric(seqs, info_gain)
        if self.strategy == "kl":
            return self.get_batch_expected_metric(seqs, kl_divergence)
        raise Exception("Unknown strategy")


    def propose(self, n_candidates=100, forbidden_seqs=[]):
        # TODO: more logic in proposing candidates
        if self.strategy == "train":
            return self.get_train_candidate(forbidden_seqs)

        # TODO: candidates can come from edits too
        candidates = []
        while len(candidates) == 0: # we don't require n_candidates?
            candidates = [self.dataset.random_seq() for _ in range(n_candidates)]
            candidates = [c for c in candidates if c not in forbidden_seqs + self.observed_seqs]

        scored_candidates = zip(candidates, self.score_candidates(candidates))
        return max(scored_candidates, key=lambda c: c[1])[0]


    def cost(self, seq):
        featurized = self.featurize(seq)
        return self.hypothesis.cost(featurized)
    

    def probs(self, seq):
        featurized = self.featurize(seq)
        return self.hypothesis.probs(featurized)
        

    def logprobs(self, seq):
        featurized = self.featurize(seq)
        return self.hypothesis.logprobs(featurized)
    
    def logits(self, seq):
        featurized = self.featurize(seq)
        return self.hypothesis.logits(featurized)
    

    def get_batch_expected_metric(self, seqs, metric):
        current_state = self.hypothesis.get_state()
        current_params = self.hypothesis.get_params()
        featurized_seqs = [self.featurize(seq) for seq in seqs]
        pos_inputs = [(self.n_features,
                       current_state,
                       torch.stack(self.observed_features + [featurized]),
                       torch.tensor(self.observed_judgments + [1.0])) 
                       for featurized in featurized_seqs]
        neg_inputs = [(self.n_features,
                       current_state,
                       torch.stack(self.observed_features + [featurized]),
                       torch.tensor(self.observed_judgments + [0.0])) 
                       for featurized in featurized_seqs]
        pos_params = self.pool.map(external_compute_params, pos_inputs)
        neg_params = self.pool.map(external_compute_params, neg_inputs)
        prob_pos = [self.hypothesis.probs(featurized) for featurized in featurized_seqs]
        expected_metrics = self.pool.map(external_expected_metric, zip(pos_params, neg_params, [current_params]*len(seqs), [metric]*len(seqs), prob_pos))
        return expected_metrics


def external_expected_metric(inputs):
    pos_params, neg_params, current_params, metric, prob_pos = inputs
    pos_delta = metric(pos_params, current_params)
    neg_delta = metric(neg_params, current_params)
    expected = prob_pos * pos_delta + (1-prob_pos) * neg_delta
    return expected


def external_compute_params(inputs):
    n_features, state, data, judgments = inputs
    s = BayesianScorer(n_features, state=state) # does not actually need state since clearing parameters before computing new
    # am i confident that pyro isn't messed up when used "as a class"? (confirm this)
    # scorer should be lightweight but maybe try to remake so this function doesn't instantiate anything (memory is main tax for multiprocessing)
    new_params, _ = s.compute_params(data, judgments, update=False)
    return new_params


def make_posterior_dist(params):
    beta = dist.Normal(params["beta_posterior_mean"], params["beta_posterior_std"])
    alpha = dist.Normal(params["alpha_posterior_mean"], params["alpha_posterior_std"])
    return beta, alpha


def entropy(params):
    beta, alpha = make_posterior_dist(params)
    return torch.sum(beta.entropy()).item() + alpha.entropy().item()


def info_gain(params1, params2):
    return entropy(params2) - entropy(params1) # reduction in entropy
    

def kl_divergence(params1, params2):
    p_beta, p_alpha = make_posterior_dist(params2)
    q_beta, q_alpha = make_posterior_dist(params1)
    kl = torch.sum(torch.distributions.kl.kl_divergence(p_beta, q_beta)).item() + torch.distributions.kl.kl_divergence(p_alpha, q_alpha).item()
    return kl
