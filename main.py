#!/usr/bin/env python3

import datasets
import informants
import learners
import scorers

import json
import numpy as np
from tqdm import tqdm

N_EVAL = 50

#@profile
def evaluate(dataset, informant, learner):
    random = np.random.RandomState(0)
    good_data = []
    bad_data = []
    for i in random.permutation(len(dataset.data)):
        datum = dataset.data[i]
        shuf_datum = dataset.permute(random, datum)
        if (datum, True) in good_data:
            continue
        if informant.judge(datum) and not informant.judge(shuf_datum):
            good_data.append((datum, True))
            bad_data.append((shuf_datum, True))
            #print(dataset.vocab.decode(datum), dataset.vocab.decode(shuf_datum))
            #print(" ".join(dataset.vocab.decode(datum)), learner.cost(datum))
            #print(" ".join(dataset.vocab.decode(shuf_datum)), learner.cost(shuf_datum))
        if len(good_data) == N_EVAL:
            break

    accepted_data = [(s, True) for s, j in learner.observations if j]
    rejected_data = [(s, True) for s, j in learner.observations if not j]

    accepted_data = accepted_data[:N_EVAL]
    rejected_data = rejected_data[:N_EVAL]

    return {
        "good": learner.full_nll(good_data),
        "bad": learner.full_nll(bad_data),
        "diff": learner.discriminate(good_data, bad_data),
        "acc": learner.full_nll(accepted_data),
        "rej": learner.full_nll(rejected_data),
    }


#@profile
def main():
    #dataset = datasets.load_cmu()
    dataset = datasets.load_cmu_onsets()
    #dataset = datasets.load_dummy()
    hw_scorer = scorers.HWScorer(dataset)
    eval_informant = informants.HWInformant(dataset, hw_scorer)
    #eval_informant = informants.DummyInformant(dataset)
    informant = eval_informant
    #informant = informants.InteractiveInformant(dataset)
    logs = {}
    #for N_INIT in [0, 8, 16, 32]:
    init_examples_all = [
        [dataset.random_example() for _ in range(128)]
        for _ in range(3)
    ]

    for N_INIT in [0, 16, 32, 64]:
        for run in range(3):
            #init_examples = [dataset.random_example() for _ in range(N_INIT)]
            #for strategy in ["train", "interleave", "std", "max", "unif"]:
            init_examples = init_examples_all[run][:N_INIT]
            for strategy in ["train", "entropy", "unif"]:
                if strategy not in logs:
                    logs[strategy] = []
                log = []
                #learner = learners.LogisticLearner(dataset, strategy=strategy)
                learner = learners.VBLearner(dataset, strategy=strategy)
                learner.initialize(n_hyps=1)
                if len(init_examples) > 0:
                    for example in init_examples[:-1]:
                        learner.observe(example, True, update=True)
                    learner.observe(init_examples[-1], True, update=True)
                scores = evaluate(dataset, eval_informant, learner)
                for k, v in scores.items():
                    print(f"{k:8s} {v:.4f}")
                print()
                for i in range(32):
                    print(strategy, run, N_INIT + i)
                    candidate = learner.propose(n_candidates=10)
                    judgment = informant.judge(candidate)
                    print(" ".join(dataset.vocab.decode(candidate)), judgment)
                    #print(learner.hypotheses[0].entropy(candidate, debug=True))
                    learner.observe(candidate, judgment)
                    #print(learner.hypotheses[0].entropy(candidate, debug=True))
                    p = learner.hypotheses[0].probs
                    print("ent", (p * np.log(p) + (1-p) * np.log(1-p)).mean())
                    scores = evaluate(dataset, eval_informant, learner)
                    for k, v in scores.items():
                        print(f"{k:8s} {v:.4f}")
                    for feat, cost in learner.top_features():
                        print(feat, cost)
                    print()
                    scores["step"] = N_INIT + i
                    scores["run"] = run
                    log.append(scores)
                logs[strategy].append(log)

                with open(f"results_{strategy}.json", "w") as writer:
                    json.dump(logs, writer)

if __name__ == "__main__":
    main()
