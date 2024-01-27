import jax
import jax.numpy as jnp
from BayesianGLM import BayesianScorer

n_features = 2

alpha_prior_mu=5.0
alpha_prior_sigma=1.0
beta_prior_mu=-10.0
beta_prior_sigma=20.0
step_size=0.01
n_updates=2000
rng_key=jax.random.PRNGKey(0)

init_params = {
    "alpha_posterior_mu": jnp.full(1, alpha_prior_mu),
    "alpha_posterior_sigma": jnp.full(1, alpha_prior_sigma),
    "beta_posterior_mu": jnp.full(n_features, beta_prior_mu),
    "beta_posterior_sigma": jnp.full(n_features, beta_prior_sigma)
}

def expected_metric(featurized_seq):
    logit = sum(featurized_seq)*beta_prior_mu + alpha_prior_mu
    prob_pos = jax.nn.sigmoid(logit)

    def compute_posterior(f, l):
        return BayesianScorer.compute_posterior(
            jnp.array([f]),
            jnp.array([l]),
            alpha_prior_mu,
            alpha_prior_sigma,
            beta_prior_mu,
            beta_prior_sigma,
            step_size,
            n_updates,
            rng_key
        )
    
    pos_params = compute_posterior(featurized_seq, 1.0)
    pos_ig = BayesianScorer.info_gain(pos_params, init_params)
    pos_kl = BayesianScorer.kl_divergence(pos_params, init_params)
    
    neg_params = compute_posterior(featurized_seq, 0.0)
    neg_ig = BayesianScorer.info_gain(neg_params, init_params)
    neg_kl = BayesianScorer.kl_divergence(neg_params, init_params)

    eig = prob_pos * pos_ig + (1 - prob_pos) * neg_ig
    ekl = prob_pos * pos_kl + (1 - prob_pos) * neg_kl
    return eig, ekl


# TEST HERE
featurized_seq = [0, 1] # binary

assert len(featurized_seq) == n_features
eig, ekl = expected_metric(featurized_seq)
print("EIG:", eig)
print("EKL:", ekl)