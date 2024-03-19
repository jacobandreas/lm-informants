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

# ------------------- SINGLE -------------------

def likelihood_given_single_constraint(features, label, constraint_idx):
    if not features[constraint_idx]:
        # if feature is off, can't say anything
        return 0.5
    return (1 - p_correct_label) if label else p_correct_label

def single_likelihood(features, label):
    return np.array([
        likelihood_given_single_constraint(features, label, i)
        for i in range(n_features)
    ])

def single_prior():
    return np.full(n_features, prior_constraint_prob)

def single_posterior(features, label):
    prior = single_prior()
    likelihood = single_likelihood(features, label)
    p_theta_and_data = likelihood * prior
    p_data = p_theta_and_data + 0.5 * (1 - prior) # likelihood is 0.5 if constraint is off
    p_theta_given_data = p_theta_and_data / p_data
    return p_theta_given_data

# ------------------- CONVERT -------------------

def joint_to_single(joint_posterior):
    matrix_dist = joint_to_matrix_dist(joint_posterior)
    single = np.array([
        np.sum(matrix_dist, axis=tuple(set(range(n_features))-{ax}))[1]
        for ax in range(n_features)
    ])
    assert np.all(single <= 1)
    assert np.all(single >= 0)
    return single

def single_to_joint(single_posterior):
    joint = np.empty(2**n_features)
    for i, active_constraints in enumerate(product([0, 1], repeat=n_features)):
        joint[i] = np.prod([
            single_posterior[j] if active_constraints[j] == 1 
            else 1 - single_posterior[j] 
            for j in range(n_features)
        ])
    assert np.allclose(np.sum(joint), 1)
    assert np.all(joint >= 0)
    return joint

def joint_to_matrix_dist(joint_posterior):
    return joint_posterior.reshape(tuple(2 for _ in range(n_features)))

def matrix_dist_to_joint(matrix_dist):
    return matrix_dist.flatten()

# ------------------- EXPECTATION -------------------

def prob_in_language(features, matrix_dist):
    # constraint must be off if feature is on
    compatible = reduce(lambda m, c: m.take([0] if c[1] else [1,0], axis=c[0]), enumerate(features), matrix_dist)
    p = np.sum(compatible)
    return p

def info_gain(posterior, prior):
    entropy_before = -np.sum(prior * np.log(prior))
    entropy_after = -np.sum(posterior * np.log(posterior))
    return entropy_before - entropy_after

def kl_divergence(posterior, prior):
    return np.sum(posterior * np.log(posterior / prior))

def expected(features, func="eig", return_all=False):
    funcs = {"eig": info_gain, "ekl": kl_divergence}
    func = funcs[func]

    prior_dist_s = joint_to_matrix_dist(single_to_joint(single_prior()))
    p_s = prob_in_language(features, prior_dist_s)
    posterior_dist_1_s = joint_to_matrix_dist(single_to_joint(single_posterior(features, 1)))
    posterior_dist_0_s = joint_to_matrix_dist(single_to_joint(single_posterior(features, 0)))
    expected_single = (func(posterior_dist_1_s, prior_dist_s) * p_s + 
                       func(posterior_dist_0_s, prior_dist_s) * (1 - p_s))
    
    prior_dist_j = joint_to_matrix_dist(joint_prior())
    p_j = prob_in_language(features, prior_dist_j)
    posterior_dist_1_j = joint_to_matrix_dist(joint_posterior(features, 1))
    posterior_dist_0_j = joint_to_matrix_dist(joint_posterior(features, 0))
    expected_joint = (func(posterior_dist_1_j, prior_dist_j) * p_j + 
                      func(posterior_dist_0_j, prior_dist_j) * (1 - p_j))
    
    if not return_all:
        return expected_single, expected_joint
    return prior_dist_s, p_s, posterior_dist_1_s, posterior_dist_0_s, expected_single, prior_dist_j, p_j, posterior_dist_1_j, posterior_dist_0_j, expected_joint

# ------------------- TESTING -------------------

def compare_dists(single_matrix, joint_matrix, rnd=5, f=None):
    expanded_single = matrix_dist_to_joint(single_matrix)
    single_posterior = joint_to_single(expanded_single)
    joint_posterior = matrix_dist_to_joint(joint_matrix)
    reduced_joint = joint_to_single(joint_posterior)

    print("Joint (by constraint):\n", np.round(reduced_joint, rnd), file=f)
    print("Single (by constraint):\n", np.round(single_posterior, rnd), file=f)
    print("Difference (by constraint):\n", np.round(reduced_joint - single_posterior, rnd), file=f)
    print(file=f)

    print("Joint (combinations):\n", np.round(joint_posterior, rnd), file=f)
    print("Single (combinations):\n", np.round(expanded_single, rnd), file=f)
    print("Difference (combinations):\n", np.round(joint_posterior - expanded_single, rnd), file=f)
    print(file=f)

def assess(features, rnd=5, f=None):
    prior_dist_s, p_s, posterior_dist_1_s, posterior_dist_0_s, eig_single, prior_dist_j, p_j, posterior_dist_1_j, posterior_dist_0_j, eig_joint = expected(features, func="eig", return_all=True)
    ekl_single, ekl_joint = expected(features, func="ekl")
    print("Priors...", file=f)
    compare_dists(prior_dist_s, prior_dist_j, rnd=rnd, f=f)

    print("Features:", features, file=f)
    print("P(in_lang) (single):", np.round(p_s, rnd), file=f)
    print("P(in_lang) (joint):", np.round(p_j, rnd), file=f)
    print(file=f)

    print("Posteriors (Y=1)...", file=f)
    compare_dists(posterior_dist_1_s, posterior_dist_1_j, rnd=rnd, f=f)
    
    print("Posteriors (Y=0)...", file=f)
    compare_dists(posterior_dist_0_s, posterior_dist_0_j, rnd=rnd, f=f)

    print("Expected Info Gain (EIG)...", file=f)
    print("Single:", np.round(eig_single, rnd), file=f)
    print("Joint:", np.round(eig_joint, rnd), file=f)
    print(file=f)

    print("Expected KL Divergence (EKL)...", file=f)
    print("Single:", np.round(ekl_single, rnd), file=f)
    print("Joint:", np.round(ekl_joint, rnd), file=f)
    print(file=f)

with open("test_0.txt", "w") as f:
    assess([0,0,0], f=f)
with open("test_1.txt", "w") as f:
    assess([1,0,0], f=f)
with open("test_2.txt", "w") as f:
    assess([1,1,0], f=f)
with open("test_3.txt", "w") as f:
    assess([1,1,1], f=f)
