#!/usr/bin/env python3

import datasets
import informants
import learners
import scorers

import json
import numpy as np
from tqdm import tqdm
#global linear_train_dataset
#global index_of_train_data_seen

N_EVAL = 50
BOUNDARY = "$"

np.random.RandomState(0)

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
def evaluate_with_external_data(good_data, bad_data, informant, learner):
    random = np.random.RandomState(0)
    #
    # good_data = []
    # for item in g:
    #     good_data.append((item,True))
    # bad_data = []
    # for item in b:
    #     bad_data.append((item,True))
    #


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

def read_in_blicks(path_to_wugs):
    intext = open(path_to_wugs,"r",encoding='utf8').read().strip().split('\n')
    return [item.split(' ') for item in intext]

#@profile
def main():
    eval_humans = False
    write_out_feat_probs = True
    dataset = datasets.load_atr_harmony()

    if write_out_feat_probs:
        feat_evals = open("FeatureProbs.csv","w",encoding="utf8")
    if eval_humans:
        dataset_to_judge_t = read_in_blicks("./data/Blicks/WordsToBeScored.csv")
        dataset_to_judge = []
        for item in dataset_to_judge_t:
            phonemes = [BOUNDARY] + item + [BOUNDARY]
            # print(phonemes,"is phonemes")
            encoded_word = dataset.vocab.encode(phonemes)  # expects a list of arpabet chars
            dataset_to_judge.append((item, encoded_word))

        good_dataset_t = read_in_blicks("./data/Blicks/hw_holdout_good.txt")
        good_dataset = []
        for item in good_dataset_t:
            # print()
            phonemes = [BOUNDARY] + item + [BOUNDARY]
            # print(phonemes,"is phonemes")
            encoded_word = dataset.vocab.encode(phonemes)  # expects a list of arpabet chars
            # print(item,encoded_word,c)
            good_dataset.append((encoded_word, True))
        bad_dataset_t = read_in_blicks("./data/Blicks/hw_holdout_bad.txt")

        bad_dataset = []
        for item in bad_dataset_t:
            # print()
            phonemes = [BOUNDARY] + item + [BOUNDARY]
            # print(phonemes,"is phonemes")
            encoded_word = dataset.vocab.encode(phonemes)  # expects a list of arpabet chars
            # print(item,encoded_word,c)
            good_dataset.append((encoded_word,True))
        #print(dataset_to_judge)
        #assert False
    linear_train_dataset = dataset.data
    np.random.shuffle(linear_train_dataset)
    print(linear_train_dataset)
    #dataset = datasets.load_cmu_onsets()
    #dataset = datasets.load_dummy()
    hw_scorer = scorers.HWScorer(dataset)
    eval_informant = informants.HWInformant(dataset, hw_scorer)
    #eval_informant = informants.DummyInformant(dataset)
    informant = eval_informant
    #informant = informants.InteractiveInformant(dataset)
    logs = {}
    if eval_humans:
        out = open("HumanEvalLogs.csv","w",encoding="utf8")
        out.write("Step,Run,Strategy,N_INIT,Item,Cost,Source\n")
    eval_metrics = open("ModelEvalLogs.csv","w",encoding="utf8")
    eval_metrics.write("ent,good,bad,diff,acc,rej,Step,Run,Strategy,N_Init\n")
    for N_INIT in [0,10,]:
        for run in range(1):
            for strategy in ["train","entropy","unif","max"]: # ,"max","unif","interleave","diff","std"
                index_of_next_item = 0
                #print(dataset.data[0])

                #print(len(dataset.data))
                init_examples = []
                for _ in range(N_INIT):
                    init_examples.append(linear_train_dataset[index_of_next_item])
                    index_of_next_item += 1
                #print("init examples are",init_examples)

                #assert False

                if strategy not in logs:
                    logs[strategy] = []
                log = []
                #learner = learners.LogisticLearner(dataset, strategy=strategy)
                learner = learners.VBLearner(dataset, strategy=strategy, linear_train_dataset = linear_train_dataset, index_of_next_item = index_of_next_item)
                learner.initialize(n_hyps=1)
                if len(init_examples) > 0:
                    for example in init_examples[:-1]:
                        learner.observe(example, True, update=True)
                    learner.observe(init_examples[-1], True, update=True)
                #scores = evaluate_with_external_data(good_dataset,bad_dataset, eval_informant, learner)
                scores = evaluate(dataset, eval_informant, learner)

                for k, v in scores.items():
                    print(f"{k:8s} {v:.4f}")
                print()
                if eval_humans:
                    for item, encoded_word in dataset_to_judge:
                        #c = learner.cost(encoded_word)
                        j = informant.cost(encoded_word)
                        # corral_of_judged_human_forms.append((item,c))
                        out.write(str(N_INIT) + ',' + str(run) + ',' + str(strategy) + ',' + str(N_INIT) + "," + str(
                            " ".join(item)) + "," + str(j) + "," + str("JUDGE") + '\n')
                        out.flush()
                for i in range(500-N_INIT):
                    print(strategy, run, N_INIT + i)
                    candidate = learner.propose(n_candidates=10)
                    judgment = informant.judge(candidate)
                    #print("candidate is", dataset.vocab.decode(candidate),"which was judged",judgment)
                    #print(" ".join(dataset.vocab.decode(candidate)), judgment)
                    #print(learner.hypotheses[0].entropy(candidate, debug=True))
                    learner.observe(candidate, judgment)
                    #print(learner.hypotheses[0].entropy(candidate, debug=True))
                    p = learner.hypotheses[0].probs
                    print("ent", (p * np.log(p) + (1-p) * np.log(1-p)).mean())
                    #scores = evaluate_with_external_data(good_dataset,bad_dataset, eval_informant, learner)
                    scores = evaluate(dataset, eval_informant, learner)
                    eval_metrics.write(str((p * np.log(p) + (1-p) * np.log(1-p)).mean())+",")
                    for k, v in scores.items():
                        print(f"{k:8s} {v:.4f}")
                        eval_metrics.write(str(v)+',')
                    eval_metrics.write(str(N_INIT+i)+','+str(run)+','+str(strategy)+','+str(N_INIT)+'\n')
                    eval_metrics.flush()

                    for feat, cost in learner.top_features():
                        print(feat, cost)
                    print()
                    if write_out_feat_probs:
                        for feat, cost in learner.all_features():
                            feat_evals.write(str(N_INIT)+","+str(feat)+','+str(cost)+","+str(N_INIT+i)+","+str(dataset.vocab.decode(candidate)).replace(",","")+","+str(judgment)+","+str(strategy)+ '\n')
                            feat_evals.flush()
                    # "ent,good,bad,diff,acc,rej,Step,Run,Strategy,N_Init\n"

                    #print("Judging human forms...")
                    #corral_of_judged_human_forms = []
                    if eval_humans:
                        for item, encoded_word in dataset_to_judge:
                            c = learner.cost(encoded_word)
                            #j = informant.cost(encoded_word)
                            #corral_of_judged_human_forms.append((item,c))
                            out.write(str(N_INIT+i)+','+str(run)+','+str(strategy)+','+str(N_INIT)+","+str(" ".join(item))+","+str(c)+","+str("LEARNER")+'\n')
                            out.flush()

                    #scores["external_wugs"] = corral_of_judged_human_forms
                    scores["step"] = N_INIT + i
                    scores["run"] = run
                    #print(t)
                    #assert False
                    log.append(scores)
                logs[strategy].append(log)

                with open(f"results_{strategy}.json", "w") as writer:
                    json.dump(logs, writer)

if __name__ == "__main__":
    main()
