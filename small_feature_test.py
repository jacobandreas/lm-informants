import numpy as np
from itertools import product
from functools import reduce

n_features = 3
prior_constraint_prob = 0.25
p_correct_label = 1-1e-10

# ------------------- JOINT -------------------

def likelihood_given_joint_constraints(features, label, active_constraints):
    in_language = (np.dot(features, active_constraints)==0).astype(int)
    return ((in_language == label) * p_correct_label +
            (in_language != label) * (1 - p_correct_label))

def joint_likelihood(features, label):
    return np.array([
        likelihood_given_joint_constraints(features, label, active_constraints)
        for active_constraints in np.array(list(product([0, 1], repeat=n_features)))
    ])

def joint_prior():
    return np.array([
        np.power(prior_constraint_prob, np.sum(constraints==1)) * np.power(1 - prior_constraint_prob, np.sum(constraints==0))
        for constraints in np.array(list(product([0, 1], repeat=n_features)))
    ])

def joint_posterior(features, label):
    prior = joint_prior()
    likelihood = joint_likelihood(features, label)
    p_theta_and_data = likelihood * prior
    p_data = np.sum(p_theta_and_data)
    p_theta_given_data = p_theta_and_data / p_data
    return p_theta_given_data

# uses likelihood function as to keep noisy label adjustment
def joint_prob_in_language(features, joint_dist):
    likelihood = joint_likelihood(features, 1)
    p = np.sum(joint_dist * likelihood)
    return p

# ------------------- SINGLE -------------------

def likelihood_given_single_constraint(features, label, constraint_idx):
    if not features[constraint_idx]:
        # if feature is off, can't say anything
        return 0.5
    return (1 - p_correct_label) if label else p_correct_label

def single_likelihood(features, label):
    if_on = np.array([
        likelihood_given_single_constraint(features, label, i)
        for i in range(n_features)
    ])
    if_off = np.array([0.5] * n_features)
    return np.stack((if_off, if_on))

def single_prior():
    return np.stack((np.full(n_features, 1 - prior_constraint_prob),
                     np.full(n_features, prior_constraint_prob)))

def single_posterior(features, label):
    prior = single_prior()
    likelihood = single_likelihood(features, label)
    p_theta_and_data = likelihood * prior
    p_data = np.sum(p_theta_and_data, axis=0)
    p_theta_given_data = p_theta_and_data / p_data
    return p_theta_given_data

# uses likelihood function as to keep noisy label adjustment
def single_prob_in_language(features, single_dist):
    likelihood = single_likelihood(features, 1)
    p = np.sum(single_dist * likelihood, axis=0)
    return p

# ------------------- CONVERT -------------------

def joint_to_single(joint_posterior):
    matrix_dist = joint_to_matrix(joint_posterior)
    single_posterior_on = np.array([
        np.sum(matrix_dist, axis=tuple(set(range(n_features))-{ax}))[1]
        for ax in range(n_features)
    ])
    assert np.all(single_posterior_on <= 1)
    assert np.all(single_posterior_on >= 0)
    single_posterior_off = 1 - single_posterior_on
    single_posterior = np.stack((single_posterior_off, single_posterior_on))
    return single_posterior

def single_to_joint(single_posterior):
    joint_posterior = np.empty(2**n_features)
    for i, active_constraints in enumerate(product([0, 1], repeat=n_features)):
        joint_posterior[i] = np.prod([
            single_posterior[active_constraints[j]][j]
            for j in range(n_features)
        ])
    assert np.allclose(np.sum(joint_posterior), 1)
    assert np.all(joint_posterior >= 0)
    return joint_posterior

def joint_to_matrix(joint_posterior):
    return joint_posterior.reshape(tuple(2 for _ in range(n_features)))

def matrix_to_joint(matrix_dist):
    return matrix_dist.flatten()

# ------------------- EXPECTATION -------------------

# "true" p(in_lang) but does not match likelihood function
def prob_in_language_OLD(features, matrix_dist):
    # constraint must be off if feature is on
    compatible = reduce(lambda m, c: m.take([0] if c[1] else [1,0], axis=c[0]), enumerate(features), matrix_dist)
    p = np.sum(compatible)
    return p

def info_gain(posterior, prior, element_wise=False):
    entropy_before = -prior * np.log(prior)
    entropy_after = -posterior * np.log(posterior)
    ig = entropy_before - entropy_after
    if element_wise:
        return ig
    return np.sum(ig)

def kl_divergence(posterior, prior, element_wise=False):
    kl = posterior * np.log(posterior / prior)
    if element_wise:
        return kl
    return np.sum(kl)

def expected(features, func="eig", return_all=False):
    funcs = {"eig": info_gain, "ekl": kl_divergence}
    func = funcs[func]

    prior_dist_s = single_prior()
    posterior_dist_1_s = single_posterior(features, 1)
    posterior_dist_0_s = single_posterior(features, 0)

    # element-wise
    p_in_lang_s = single_prob_in_language(features, prior_dist_s)
    expected_s = (func(posterior_dist_1_s, prior_dist_s, element_wise=True) * p_in_lang_s +
                  func(posterior_dist_0_s, prior_dist_s, element_wise=True) * (1 - p_in_lang_s))
    
    # as currently done
    p_in_lang_s = joint_prob_in_language(features, single_to_joint(prior_dist_s))
    expected_s_broken = (func(posterior_dist_1_s, prior_dist_s, element_wise=False) * p_in_lang_s +
                         func(posterior_dist_0_s, prior_dist_s, element_wise=False) * (1 - p_in_lang_s))


    prior_dist_j = joint_prior()
    p_in_lang_j = joint_prob_in_language(features, prior_dist_j)
    posterior_dist_1_j = joint_posterior(features, 1)
    posterior_dist_0_j = joint_posterior(features, 0)
    expected_j = (func(posterior_dist_1_j, prior_dist_j) * p_in_lang_j +
                  func(posterior_dist_0_j, prior_dist_j) * (1 - p_in_lang_j))

    if return_all:
        return prior_dist_s, p_in_lang_s, posterior_dist_1_s, posterior_dist_0_s, expected_s, expected_s_broken, prior_dist_j, p_in_lang_j, posterior_dist_1_j, posterior_dist_0_j, expected_j
    return expected_s, expected_s_broken, expected_j

# ------------------- TESTING -------------------

def compare_dists(single_dist, joint_dist, rnd=5, f=None):
    single_as_joint = single_to_joint(single_dist)
    joint_as_single = joint_to_single(joint_dist)[1]
    single_dist = single_dist[1] # only show p(on)

    print("Joint (by constraint):\n", np.round(joint_as_single, rnd), file=f)
    print("Single (by constraint):\n", np.round(single_dist, rnd), file=f)
    print("Difference (by constraint):\n", np.round(joint_as_single - single_dist, rnd), file=f)
    print(file=f)

    print("Joint (combinations):\n", np.round(joint_dist, rnd), file=f)
    print("Single (combinations):\n", np.round(single_as_joint, rnd), file=f)
    print("Difference (combinations):\n", np.round(joint_dist - single_as_joint, rnd), file=f)
    print(file=f)

def assess(features, rnd=5, f=None):
    prior_dist_s, p_in_lang_s, posterior_dist_1_s, posterior_dist_0_s, eig_s, eig_s_broken, prior_dist_j, p_in_lang_j, posterior_dist_1_j, posterior_dist_0_j, eig_j = expected(features, func="eig", return_all=True)
    ekl_s, ekl_s_broken, ekl_j = expected(features, func="ekl")

    print("Priors...", file=f)
    compare_dists(prior_dist_s, prior_dist_j, rnd=rnd, f=f)

    print("Features:", features, file=f)
    print("P(in_lang) (single):", np.round(p_in_lang_s, rnd), file=f)
    print("P(in_lang) (joint):", np.round(p_in_lang_j, rnd), file=f)
    print(file=f)

    print("Posteriors (Y=1)...", file=f)
    compare_dists(posterior_dist_1_s, posterior_dist_1_j, rnd=rnd, f=f)
    
    print("Posteriors (Y=0)...", file=f)
    compare_dists(posterior_dist_0_s, posterior_dist_0_j, rnd=rnd, f=f)

    print("Expected Info Gain (EIG)...", file=f)
    print("Single (as currently done):", np.round(eig_s_broken, rnd), file=f)
    print("Single:", np.round(np.sum(eig_s), rnd), file=f)
    print("Joint:", np.round(eig_j, rnd), file=f)
    print(file=f)

    print("Expected KL Divergence (EKL)...", file=f)
    print("Single (as currently done):", np.round(ekl_s_broken, rnd), file=f)
    print("Single:", np.round(np.sum(ekl_s), rnd), file=f)
    print("Joint:", np.round(ekl_j, rnd), file=f)
    print(file=f)

with open("test_0.txt", "w") as f:
    assess([0,0,0], f=f)
with open("test_1.txt", "w") as f:
    assess([1,0,0], f=f)
with open("test_2.txt", "w") as f:
    assess([1,1,0], f=f)
with open("test_3.txt", "w") as f:
    assess([1,1,1], f=f)
