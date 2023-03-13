#!/usr/bin/env python3

import datasets
import os
import informants
import learners
import scorers
import argparse
import pandas as pd
import json
import numpy as np
import itertools as it
from tqdm import tqdm
from AnalyzeSyntheticData import is_ti
import wandb
import matplotlib.pyplot as plt

from util import entropy
import csv

N_EVAL = 50
BOUNDARY = "$"

np.random.seed(0)

#@profile

def get_broad_annotations():
    df = pd.read_csv("broad_test_set_annotated.csv") 
    return dict(zip(df.Word, df.IsLicit)), dict(zip(df.Word, df.IsTI))

def plot_feature_probs(features, costs, last_costs, step):

    fig = plt.figure(figsize=(12, 3))

    colors = ["blue" if c == lc else "red" for c, lc in zip(costs, last_costs)] 

    # Create the plot
    plt.clf()
    plt.scatter(features, costs, linestyle='None', marker='o', c=colors, s=3, alpha=0.5)
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.ylim(-0.1, 1.1)
    plt.ylabel('Prob')
    plt.rc('xtick',labelsize=5)
    plt.title('Prob vs Feature for Step {}'.format(step))
    return fig

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

    accepted_data = [(s, True) for s, _, j in learner.observations if j]
    rejected_data = [(s, True) for s, _, j in learner.observations if not j]

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


    accepted_data = [(s, True) for s, _, j in learner.observations if j]
    rejected_data = [(s, True) for s, _, j in learner.observations if not j]

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

def get_out_file(file_name, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return open(os.path.join(out_dir, file_name), "w", encoding='utf8', buffering=1)

def get_csv_writer(file_name, out_dir):
    f = get_out_file(file_name, out_dir)
    writer = csv.writer(f, delimiter=",")
    return writer 

#@profile
def main(args):
    list_of_words_to_get_features_from = open("all_sylls.csv","r").read().split('\n')
    list_of_words_to_get_features_from = [i for i in list_of_words_to_get_features_from if i]
    print(list_of_words_to_get_features_from)
    #assert False
    write_out_all_features_first_time = False # don't touch this
    eval_humans = True
    write_out_feat_probs = True
    get_prior_prob_of_test_set = True
    feature_query_log = get_out_file("feature_query_log.csv", args.exp_dir) 
    feature_query_log.write("Feature,Candidate,Step,N_Init,Strategy,Run\n")
    alL_features_log = get_out_file("all_features_log.csv", args.exp_dir) 
#    temp_dataset = datasets.load_atr_harmony()
#    dataset = datasets.load_manual()
#    dataset.vocab = temp_dataset.vocab
    dataset = datasets.load_atr_harmony()

    random = np.random.RandomState(0)
    if get_prior_prob_of_test_set:
        prior_probs_writer = get_csv_writer("prior_probabilities_of_test_set_items.csv", args.exp_dir)
        prior_probs_writer.writerow(["Word", "ProbAcceptable"])
    if write_out_feat_probs:
        feat_evals = get_out_file("FeatureProbs.csv", args.exp_dir)
        feat_evals.write("N_Init,feature,cost,Step,Candidate,Judgment,Strategy, IsTI,Run\n")
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

        broad_licit_annotations, broad_TI_annotations = get_broad_annotations()
      
        
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
    unique_features = pd.read_csv('all_features_in_data_unique.csv')['X1'].unique()
    last_costs = [None] * len(unique_features)

    if eval_humans:
        out_human_evals = get_out_file("HoldoutEvals.csv", args.exp_dir) 
        out_human_evals.write("Step,Run,Strategy,N_INIT,Item,Cost,Source,TestType\n")
        broad_human_evals_writer = get_csv_writer("BroadHoldoutEvals.csv", args.exp_dir) 
        broad_human_evals_writer.writerow(["Step", "Run", "Strategy", "N_INIT", "Items", "Costs", "IsLicit", "IsTI"])
    eval_metrics = get_out_file("ModelEvalLogs.csv", args.exp_dir)
    eval_metrics.write("ent,good,bad,diff,acc,rej,Step,Run,Strategy,N_Init,IsTI,judgement,proposed_form,entropy_before,entropy_after,entropy_diff,change_in_probs\n") # including things it queried about
    results_by_observations_writer = get_csv_writer("ResultsByObservations.csv", args.exp_dir)
    results_by_observations_writer.writerow(["Step", "Run", "strategy", "candidate", "judgment", "new_probs", "p_all_off", "update_unclipped", "update_clipped"])

    if args.do_plot_wandb:
        import wandb

    for N_INIT in [0]:
        num_runs = 2 
        for run in range(num_runs):
            #for strategy in ["train","entropy","unif","max","std","diff"]: # ,"max","unif","interleave","diff","std"
#            for strategy in ["", "eig", "unif","train"]: # only train, entropy, eig, and unif are well-defined here
            for strategy in ["entropy", "train", "unif"]: # only train, entropy, eig, and unif are well-defined here
                print("STRATEGY:", strategy)
                if args.do_plot_wandb:
                    config = {"n_init": N_INIT, "run": run, "strategy": strategy, "log_log_alpha_ratio": args.log_log_alpha_ratio, "prior_prob": args.prior_prob}
                    wandb_run = wandb.init(config=config, project=args.wandb_project, name=strategy, reinit=True)
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
                learner.initialize(n_hyps=1, log_log_alpha_ratio=args.log_log_alpha_ratio, prior_prob=args.prior_prob)
                if len(init_examples) > 0:
                    for example in init_examples[:-1]:
                        learner.observe(example, True, update=True, verbose=args.verbose, batch=args.batch)
                    learner.observe(init_examples[-1], True, update=True, verbose=args.verbose)
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
                        prior_probs_writer.writerow([str(broad_test_set_t[i]).replace(","," "), str(c)])
                    get_prior_prob_of_test_set = False
                #assert False


#                for i in range(75-N_INIT):
                for i in range(75-N_INIT):
                    print("")
                    print(f"i: {i}")
                    step=N_INIT+i
                    p = learner.hypotheses[0].probs

                    if args.do_plot_wandb:
                        all_features = [(c, f) for (c, f) in learner.all_features() if f in unique_features]
                        all_features.sort(key=lambda x: x[1])
                        costs = [x[0] for x in all_features]
                        features = [x[1] for x in all_features]
                        feature_probs_plot = plot_feature_probs(features, costs, last_costs, step)
                        last_costs = costs.copy()
                        wandb.log({"feature_probs/plot": wandb.Image(feature_probs_plot)})
                    #learner.cost()
                    candidate = learner.propose(n_candidates=100, forbidden_data = forbidden_data_that_cannot_be_queried_about, length_norm=True)
                    judgment = informant.judge(candidate)
                    #prior_probability_of_accept = learner.cost(candidate)


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
                        feature_query_log.write(str(thing)+','+str(dataset.vocab.decode(candidate))+","+str(step)+","+str(N_INIT)+","+str(strategy)+","+str(run)+"\n")
                    if write_out_all_features_first_time == False:
                        for thing in total_features:
                            alL_features_log.write(str(thing)+"\n")
                            write_out_all_features_first_time = True

                    #print(len(total_features),total_features)
                    #print("____________")
                    if eval_humans:
                        for item, encoded_word in narrow_test_set:
                            c = learner.cost(encoded_word)
                            #j = informant.cost(encoded_word)
                            #corral_of_judged_human_forms.append((item,c))
                            str_item = " ".join(item)
                            out_human_evals.write(str(step)+','+str(run)+','+str(strategy)+','+str(N_INIT)+","+str_item+","+str(c)+","+str("LEARNER")+",NarrowTest"+'\n')
                            out_human_evals.flush()

                        items, labels, TIs, costs = [], [], [], []
                        for item_idx, (item, encoded_word) in enumerate(broad_test_set):
                            c = learner.cost(encoded_word)
                            if item_idx == 0:
                                print(f"item {item_idx}: {item}; cost: {c}")
                            costs.append(c)
                            #j = informant.cost(encoded_word)
                            #corral_of_judged_human_forms.append((item,c))
                            str_item = " ".join(item)
                            out_human_evals.write(str(step)+','+str(run)+','+str(strategy)+','+str(N_INIT)+","+str_item+","+str(c)+","+str("LEARNER")+",BroadTest"+'\n')
                            isLicit = int(broad_licit_annotations[str_item])
                            isTI = int(broad_TI_annotations[str_item])
                            labels.append(isLicit)
                            items.append(str_item)
                            TIs.append(isTI)
                            
                            out_human_evals.flush()
                        broad_human_evals_writer.writerow([step, run, strategy, N_INIT, items, costs, labels, TIs])
                        print(f"avg cost for broad test set: {np.mean(costs)}")

                    #scores["external_wugs"] = corral_of_judged_human_forms
                    scores["step"] = N_INIT + i
                    scores["run"] = run
                    #print(t)
                    #assert False
                    log.append(scores)
                    print()
                    print(f"strategy: {strategy}, run: {run}/{num_runs}, step: {N_INIT + i}")
                    
                    #for feat, cost in learner.top_features():
                    #    print(feat, cost)
                    #print()
                    if write_out_feat_probs:
                        for cost, feat in learner.all_features():
                            feat_evals.write(str(N_INIT)+","+str(feat)+','+str(cost)+","+str(step)+","+str(dataset.vocab.decode(candidate)).replace(",","")+","+str(judgment)+","+str(strategy)+","+str(is_ti(str(dataset.vocab.decode(candidate)).replace(",","")))+','+str(run)+ '\n')
                            
                            feat_evals.flush()

                    # "ent,good,bad,diff,acc,rej,Step,Run,Strategy,N_Init\n"
                    
                    entropy_before = entropy(learner.hypotheses[0].probs)
                    probs_before = learner.hypotheses[0].probs.copy()
                    learner.observe(candidate, judgment, verbose=args.verbose)
                    last_result = learner.results_by_observations[-1] 
                    last_result_DL = {k: [dic[k] for dic in last_result] for k in last_result[0]}
                    results_by_observations_writer.writerow([i, run, strategy, dataset.vocab.decode(candidate), judgment, last_result_DL["new_probs"], last_result_DL["p_all_off"], last_result_DL["update_unclipped"], last_result_DL["update_clipped"]])
                    
                    probs_after = learner.hypotheses[0].probs.copy()
                    entropy_after = entropy(learner.hypotheses[0].probs)
                    change_in_probs = np.linalg.norm(probs_after-probs_before)
                    print("entropy before: ", entropy_before) 
                    print("entropy after: ", entropy_after)
                    entropy_diff = entropy_before-entropy_after
                    print("information gain:", entropy_diff)
                    print("change in probs (norm):", change_in_probs)
                    prob_positive = np.exp(learner.hypotheses[0].logprob(candidate, True))
                    prob_negative = np.exp(learner.hypotheses[0].logprob(candidate, False))
                    print("prob positive:", prob_positive)
                    print("prob negative:", prob_negative)

                    eval_metrics.write(str((p * np.log(p) + (1-p) * np.log(1-p)).mean())+",")
                    for k, v in scores.items():
                        #print(f"{k:8s} {v:.4f}")
                        eval_metrics.write(str(v)+',')
                    eval_metrics.write(str(step)+','+str(run)+','+str(strategy)+','+str(N_INIT)+","+str(is_ti(str(dataset.vocab.decode(candidate)).replace(",","")))+","+str(judgment)+","+str(dataset.vocab.decode(candidate)).replace(",","")+","+str(entropy_before)+","+str(entropy_after)+","+str(entropy_diff)+","+str(change_in_probs)+'\n')
                    eval_metrics.flush()

#
                    #print("Judging human forms...")
                    #corral_of_judged_human_forms = []

                    # print(learner.hypotheses[0].entropy(candidate, debug=True))
                logs[strategy].append(log)

#                with open(f"results_{strategy}.json", "w") as writer:
#                    json.dump(logs, writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--log_log_alpha_ratio", type=float, default=1)
    parser.add_argument("--prior_prob", type=float, default=0.5)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    
    # batch defaults to True
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--no-batch', dest='batch', action='store_false')
    parser.set_defaults(batch=True)

    args = parser.parse_args()

    if args.wandb_project is not None:
        args.do_plot_wandb = True
    else:
        args.do_plot_wandb=False

    print("args: ", args)

    main(args)
