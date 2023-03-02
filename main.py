#!/usr/bin/env python3

import datasets
import informants
import learners
import scorers

import json
import numpy as np
import itertools as it
from tqdm import tqdm
from AnalyzeSyntheticData import is_ti

from util import entropy

N_EVAL = 50
BOUNDARY = "$"

np.random.seed(0)

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
    #print("returning blicks", intext,"from path",path_to_wugs)
    return [item.split(' ') for item in intext]

#@profile
def main():
    list_of_words_to_get_features_from = open("all_sylls.csv","r").read().split('\n')
    list_of_words_to_get_features_from = [i for i in list_of_words_to_get_features_from if i]
    print(list_of_words_to_get_features_from)
    #assert False
    write_out_all_features_first_time = False # don't touch this
    eval_humans = True
    write_out_feat_probs = True
    get_prior_prob_of_test_set = True
    feature_query_log = open("feature_query_log.csv","w",encoding='utf8')
    feature_query_log.write("Feature,Candidate,Step,N_Init,Strategy,Run\n")
    alL_features_log = open("all_features_log.csv","w",encoding='utf8')
    dataset = datasets.load_atr_harmony()

    random = np.random.RandomState(0)
    if get_prior_prob_of_test_set:
        prior_probs = open("prior_probabilities_of_test_set_items.csv","w",encoding="utf8")
        prior_probs.write("Word,ProbAcceptable\n")
    if write_out_feat_probs:
        feat_evals = open("FeatureProbs.csv","w",encoding="utf8")
        feat_evals.write("N_Init, feature, cost, Step, Candidate, Judgment, Strategy, IsTI, Run\n")
    if eval_humans:
        narrow_test_set_t = read_in_blicks("TI_test.csv")
        narrow_test_set = []
        for item in narrow_test_set_t:
            phonemes = [BOUNDARY] + item + [BOUNDARY]
            #print(phonemes,"is phonemes")
            encoded_word = dataset.vocab.encode(phonemes)  # expects a list of arpabet chars
            narrow_test_set.append((item, encoded_word))

        broad_test_set_t = read_in_blicks("test_set.csv")
        broad_test_set = []
        for item in broad_test_set_t:
            phonemes = [BOUNDARY] + item + [BOUNDARY]
            # print(phonemes,"is phonemes")
            encoded_word = dataset.vocab.encode(phonemes)  # expects a list of arpabet chars
            broad_test_set.append((item, encoded_word))
        # good_dataset_t = read_in_blicks("./data/Blicks/hw_holdout_good.txt")
        # good_dataset = []
        # for item in good_dataset_t:
        #     # print()
        #     phonemes = [BOUNDARY] + item + [BOUNDARY]
        #     # print(phonemes,"is phonemes")
        #     encoded_word = dataset.vocab.encode(phonemes)  # expects a list of arpabet chars
        #     # print(item,encoded_word,c)
        #     good_dataset.append((encoded_word, True))
        # bad_dataset_t = read_in_blicks("./data/Blicks/hw_holdout_bad.txt")
        #
        # bad_dataset = []
        # for item in bad_dataset_t:
        #     # print()
        #     phonemes = [BOUNDARY] + item + [BOUNDARY]
        #     # print(phonemes,"is phonemes")
        #     encoded_word = dataset.vocab.encode(phonemes)  # expects a list of arpabet chars
        #     # print(item,encoded_word,c)
        #     good_dataset.append((encoded_word,True))
        #print(dataset_to_judge)
        #assert False
    forbidden_data_that_cannot_be_queried_about = [item[1] for item in broad_test_set] + [item[1] for item in narrow_test_set] #broad_test_set+narrow_test_set
    linear_train_dataset = dataset.data
    random.shuffle(linear_train_dataset)
    #print(linear_train_dataset)
    #dataset = datasets.load_cmu_onsets()
    #dataset = datasets.load_dummy()
    hw_scorer = scorers.HWScorer(dataset)
    eval_informant = informants.HWInformant(dataset, hw_scorer)
    #eval_informant = informants.DummyInformant(dataset)
    informant = eval_informant
    #informant = informants.InteractiveInformant(dataset)

    logs = {}
    if eval_humans:
        out_human_evals = open("HoldoutEvals.csv","w",encoding="utf8")
        out_human_evals.write("Step,Run,Strategy,N_INIT,Item,Cost,Source,TestType\n")
    eval_metrics = open("ModelEvalLogs.csv","w",encoding="utf8")
    eval_metrics.write("ent,good,bad,diff,acc,rej,Step,Run,Strategy,N_Init,IsTI,judgement,proposed_form\n") # including things it queried about
    for N_INIT in [0]:
        num_runs = 1 
        for run in range(num_runs):
            #for strategy in ["train","entropy","unif","max","std","diff"]: # ,"max","unif","interleave","diff","std"
#            for strategy in ["", "eig", "unif","train"]: # only train, entropy, eig, and unif are well-defined here
            for strategy in ["eig"]: # only train, entropy, eig, and unif are well-defined here
                print("STRATEGY:", strategy)
                #if strategy == "train":
                #    run = 19
                index_of_next_item = 0
                #print(dataset.data[0])
                random.shuffle(linear_train_dataset) # turn this off if you want to have train be always the same order.

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

                #for k, v in scores.items():
                    #print(f"{k:8s} {v:.4f}")
                print()
                # if eval_humans:
                #     for item, encoded_word in dataset_to_judge:
                #         #c = learner.cost(encoded_word)
                #         j = informant.cost(encoded_word)
                #         # corral_of_judged_human_forms.append((item,c))
                #         out.write(str(N_INIT) + ',' + str(run) + ',' + str(strategy) + ',' + str(N_INIT) + "," + str(
                #             " ".join(item)) + "," + str(j) + "," + str("JUDGE") + '\n')
                #         out.flush()

                # get prior prob of item in heldout dataset
                if get_prior_prob_of_test_set:
                    for i in range(len(broad_test_set)):
                        #print(broad_test_set[i])# encoded item
                        #print(broad_test_set_t[i])# human readible item
                        c = np.exp(learner.hypotheses[0].logprob(broad_test_set[i][1],True,length_norm=False))
# this can't be, b/c we're interested in the actual prob
                        prior_probs.write(str(broad_test_set_t[i]).replace(","," ")+','+str(c)+'\n')
                    get_prior_prob_of_test_set = False
                #assert False


#                for i in range(75-N_INIT):
                for i in range(20-N_INIT):
                    print("")
                    print(f"i: {i}")
                    #learner.cost()
                    candidate = learner.propose(n_candidates=100, forbidden_data = forbidden_data_that_cannot_be_queried_about, length_norm=True)
                    judgment = informant.judge(candidate)
                    #prior_probability_of_accept = learner.cost(candidate)

                    entropy_before = entropy(learner.hypotheses[0].probs)
                    learner.observe(candidate, judgment)
                    entropy_after = entropy(learner.hypotheses[0].probs)
                    print("entropy before: ", entropy_before) 
                    print("entropy after: ", entropy_after) 
                    print("information gain:", entropy_before-entropy_after)
                    prob_positive = np.exp(learner.hypotheses[0].logprob(candidate, True))
                    prob_negative = np.exp(learner.hypotheses[0].logprob(candidate, False))
                    print("prob positive:", prob_positive)
                    print("prob negative:", prob_negative)

                    p = learner.hypotheses[0].probs

                    #print("ent", (p * np.log(p) + (1-p) * np.log(1-p)).mean())
                    #scores = evaluate_with_external_data(good_dataset,bad_dataset, eval_informant, learner)
                    scores = evaluate(dataset, eval_informant, learner)
                    #print(dataset.vocab.decode(candidate))


                    total_features = []
                    total_features_2 = []
                    mean_field_scorer = learner.hypotheses[0]
                    features2 = mean_field_scorer._featurize(candidate).nonzero()[0]

                    # UNCOMMENT UNTIL ASSERT FALSE TO FIND OUT HOW MANY FEATURES THERE ARE IN THE DATA!!
                    # a = open("all_feats_in_data.csv","w")
                    # for cand in list_of_words_to_get_features_from:
                    #     phonemes = [BOUNDARY] + cand.split(" ") + [BOUNDARY]
                    #     # print(phonemes,"is phonemes")
                    #     encoded_word = dataset.vocab.encode(phonemes)
                    #     for z in range(len(encoded_word) - mean_field_scorer.ORDER + 1):
                    #         features_here = [mean_field_scorer.phoneme_features[encoded_word[j]].nonzero()[0] for j in
                    #                          range(z, z + mean_field_scorer.ORDER)]
                    #         for ff in it.product(*features_here):
                    #             # features[mean_field_scorer.ngram_features[ff]] += 1
                    #             parts = " :: ".join(mean_field_scorer.feature_vocab.get_rev(f) for f in ff)
                    #
                    #             print(parts)
                    #
                    #             a.write(parts+"\n")
                    #
                    # assert False
                    #print("feat specific locally",features2)
                    #features = np.zeros(len(mean_field_scorer.ngram_features))
                    for z in range(len(candidate) - mean_field_scorer.ORDER + 1):
                        features_here = [mean_field_scorer.phoneme_features[candidate[j]].nonzero()[0] for j in range(z, z + mean_field_scorer.ORDER)]
                        for ff in it.product(*features_here):
                            #features[mean_field_scorer.ngram_features[ff]] += 1
                            parts = " :: ".join(mean_field_scorer.feature_vocab.get_rev(f) for f in ff)
                            total_features_2.append(parts)
                    #print(features)

                    #h2 = learner.hypotheses[0]._featurize(candidate)
                    #h3 = learner.hypotheses[0]._featurize(candidate).nonzero()

                    #h = learner.hypotheses[0]._featurize(candidate).nonzero()[0]
                    for q, ngram_feat in enumerate(mean_field_scorer.ngram_features.keys()):
                        parts = " :: ".join(mean_field_scorer.feature_vocab.get_rev(f) for f in ngram_feat)
                        total_features.append((mean_field_scorer.probs[q].item(), parts))
                    #for ngram_feat in features2:
                    #    parts = " :: ".join(mean_field_scorer.feature_vocab.get_rev(f) for f in ngram_feat)
                    #    total_features_2.append((mean_field_scorer.probs[i].item(), parts))
                    #print("here are the features active in ",dataset.vocab.decode(candidate))
                    #print(len(total_features_2),total_features_2)
                    #print("here are all the features")
                    for thing in total_features_2:
                        feature_query_log.write(str(thing)+','+str(dataset.vocab.decode(candidate))+","+str(N_INIT+i)+","+str(N_INIT)+","+str(strategy)+","+str(run)+"\n")
                    if write_out_all_features_first_time == False:
                        for thing in total_features:
                            alL_features_log.write(str(thing)+"\n")
                            write_out_all_features_first_time = True

                    #print(len(total_features),total_features)
                    #print("____________")
                    eval_metrics.write(str((p * np.log(p) + (1-p) * np.log(1-p)).mean())+",")
                    for k, v in scores.items():
                        #print(f"{k:8s} {v:.4f}")
                        eval_metrics.write(str(v)+',')
                    eval_metrics.write(str(N_INIT+i)+','+str(run)+','+str(strategy)+','+str(N_INIT)+","+str(is_ti(str(dataset.vocab.decode(candidate)).replace(",","")))+","+str(judgment)+","+str(dataset.vocab.decode(candidate)).replace(",","")+'\n')
                    eval_metrics.flush()

                    #for feat, cost in learner.top_features():
                    #    print(feat, cost)
                    #print()
                    if write_out_feat_probs:
                        for cost, feat in learner.all_features():
                            feat_evals.write(str(N_INIT)+","+str(feat)+','+str(cost)+","+str(N_INIT+i)+","+str(dataset.vocab.decode(candidate)).replace(",","")+","+str(judgment)+","+str(strategy)+","+str(is_ti(str(dataset.vocab.decode(candidate)).replace(",","")))+','+str(run)+ '\n')
                            feat_evals.flush()
                    # "ent,good,bad,diff,acc,rej,Step,Run,Strategy,N_Init\n"
#
                    #print("Judging human forms...")
                    #corral_of_judged_human_forms = []
                    if eval_humans:
                        for item, encoded_word in narrow_test_set:
                            c = learner.cost(encoded_word)
                            #j = informant.cost(encoded_word)
                            #corral_of_judged_human_forms.append((item,c))
                            out_human_evals.write(str(N_INIT+i)+','+str(run)+','+str(strategy)+','+str(N_INIT)+","+str(" ".join(item))+","+str(c)+","+str("LEARNER")+",NarrowTest"+'\n')
                            out_human_evals.flush()

                        for item, encoded_word in broad_test_set:
                            c = learner.cost(encoded_word)
                            #j = informant.cost(encoded_word)
                            #corral_of_judged_human_forms.append((item,c))
                            out_human_evals.write(str(N_INIT+i)+','+str(run)+','+str(strategy)+','+str(N_INIT)+","+str(" ".join(item))+","+str(c)+","+str("LEARNER")+",BroadTest"+'\n')
                            out_human_evals.flush()

                    #scores["external_wugs"] = corral_of_judged_human_forms
                    scores["step"] = N_INIT + i
                    scores["run"] = run
                    #print(t)
                    #assert False
                    log.append(scores)
                    print()
                    print(f"strategy: {strategy}, run: {run}/{num_runs}, step: {N_INIT + i}")


                    # print(learner.hypotheses[0].entropy(candidate, debug=True))
                logs[strategy].append(log)

                with open(f"results_{strategy}.json", "w") as writer:
                    json.dump(logs, writer)

if __name__ == "__main__":
    main()
