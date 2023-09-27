#!/usr/bin/env python3

import datasets
from scipy import stats
import wandb
from sklearn import metrics
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
import time
import multiprocessing

import cProfile
import pdb

from util import entropy, plot_feature_probs, kl_bern
import csv

N_EVAL = 50
BOUNDARY = "$"

np.random.seed(0)

def cost_to_prob(cost):
    return np.exp(cost * -1)

#@profile
def eval_auc(costs, labels):
    probs = [cost_to_prob(c) for c in costs]
    labels = [int(l) for l in labels]
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def eval_corrs(costs, labels, sources, items, num_phonemes, num_features): # nb, sources comes in via TIs, and labels are human judgments. now, if soucres = 5, these are pulled off, and used for auc calculation
    data = pd.DataFrame({'costs': costs, 'labels': labels, 'sources': sources, 'items':items, 'num_phonemes': num_phonemes, 'num_features': num_features})
    '''
    print("eval corrs breakdown:")
    print(data['sources'].value_counts())
    '''
    # Select rows with sources 1, 2, 3, or 4
    df1 = data.loc[data['sources'].isin([1, 2, 3, 4])]

    # Select rows with source 5
    df2 = data.loc[data['sources'] == 5]

#    group_corr = df1.groupby('sources').apply(lambda x: np.corrcoef(-x['costs'], x['labels'])[0, 1]).reset_index(        name='spearman_corr')
    group_corr = df1.groupby('sources').apply(lambda x: stats.spearmanr(-x['costs'], x['labels'])[0]).reset_index(name='spearman_corr')
    print("CORR:")
    #print(group_corr)

    # Extract 'costs' and 'labels' columns from df2 as lists
    
    costs = df2['costs'].tolist()
    labels = df2['labels'].tolist()
    roc_auc = eval_auc(costs, labels)
    auc_results = {'auc': roc_auc}
    print("roc_auc is", roc_auc)

    for num in df2['num_phonemes'].unique(): 
        temp = df2[df2['num_phonemes']==num]
#        print(f'num phonemes = {num} ({len(temp)})')
#        print(temp.head(3))
        costs = temp['costs'].tolist()
        labels = temp['labels'].tolist()
        roc_auc = eval_auc(costs, labels)
        auc_results[f'auc_{num}_features']=roc_auc

    print(group_corr)
    print("number of forms in auc test set is",len(df2))
    '''
    print("top 10:")
    print(df2.head(10))
    print("10 randomly sampled: ")
    print(df2.sample(10))
    
    print("number of forms in human test set is",len(df1))
    print("top 10:")
    print(df1.head(10))
    print("10 randomly sampled: ")
    print(df1.sample(10))
    
    print("number of forms in full test set is",len(data))
#    pdb.set_trace()
    '''
    return group_corr, auc_results, df2


def get_broad_annotations(feature_type):
    if feature_type == "atr_harmony":
        df = pd.read_csv("broad_test_set_annotated.csv")
        return dict(zip(df.Word, df.IsLicit)), dict(zip(df.Word, df.IsTI))
    elif feature_type == "english":
        df = pd.read_csv("WordsAndScoresFixed_newest.csv")
        df['Word'] = df.apply(lambda row: row['Word'].strip(), axis=1)
        assert df['Word'].value_counts().max() == 1, (f'Repeats of words found in dataset, '
                'which will lead to overriding of annotations (which are dicts)')
        return dict(zip(df.Word, df.Score)), dict(zip(df.Word, df.Source))
    else:
        raise NotImplementedError("please select a valid feature type!")


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
#    print(list_of_words_to_get_features_from)
    #assert False
    write_out_all_features_first_time = False # don't touch this
    eval_humans = args.eval_humans 
    write_out_feat_probs = True
    get_prior_prob_of_test_set = False 
    feature_query_log = get_out_file("feature_query_log.csv", args.exp_dir) 
    feature_query_log.write("Feature,Candidate,Step,N_Init,Strategy,Run\n")
    alL_features_log = get_out_file("all_features_log.csv", args.exp_dir) 

    if args.lexicon_file is None:
        lexicon_file_name = f"data/hw/{args.feature_type}_lexicon.txt"
    else:
        lexicon_file_name = args.lexicon_file
        assert os.path.exists(lexicon_file_name) 
    dataset = datasets.load_lexicon(lexicon_file_name, min_length=args.min_length, max_length=args.max_length)

    if args.feature_type == "english":
        filtered_features = pd.read_csv('all_feats_in_data_english.csv', header=None)[0].values
        print(filtered_features)
        print(len(filtered_features))
    else:
        filtered_features = None

    random = np.random.RandomState(0)
    mean_field_scorer = scorers.MeanFieldScorer(dataset, feature_type=args.feature_type, features=filtered_features)
    print("len(probs):", (mean_field_scorer.probs).shape)
    if get_prior_prob_of_test_set:
        prior_probs_writer = get_csv_writer("prior_probabilities_of_test_set_items.csv", args.exp_dir)
        prior_probs_writer.writerow(["Word", "ProbAcceptable"])
    if write_out_feat_probs:
        feat_evals = get_out_file("FeatureProbs.csv", args.exp_dir)
        feat_evals.write("N_Init,feature,cost,Step,Candidate,Judgment,Strategy, IsTI,Run\n")
    if eval_humans and args.feature_type == "atr_harmony":
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
            featurized = mean_field_scorer._featurize(encoded_word).nonzero()
            broad_test_set.append((item, encoded_word, featurized))

        broad_licit_annotations, broad_TI_annotations = get_broad_annotations(args.feature_type)
    elif eval_humans and args.feature_type == "english":
        #_t = read_in_blicks("TI_test.csv")
        narrow_test_set = []

#        broad_test_set_t = read_in_blicks("WordsToBeScored.csv")
        raw_eval_dataset = pd.read_csv('WordsAndScoresFixed_newest.csv')

        items = [i.strip().split(' ') for i in raw_eval_dataset['Word'].values]
        sources = [int(s) for s in raw_eval_dataset['Source'].values]
        labels = raw_eval_dataset['Score'].values
        encoded_items = []
        num_phonemes = []
        num_features = []
        featurized_items = []
        print("Reading test set...")
        assert len(items) == len(sources)
        for item, source in tqdm(zip(items, sources), total=len(items)):
            num_phonemes.append(len(item))
            if str(source).strip() == '1':
                phonemes = [BOUNDARY] + item
            else:
                phonemes = [BOUNDARY] + item + [BOUNDARY]
            encoded_item = dataset.vocab.encode(phonemes)  # expects a list of arpabet chars
            encoded_items.append(encoded_item)

            featurized = mean_field_scorer._featurize(encoded_item).nonzero()
            num_features.append(len(featurized))
            featurized_items.append(featurized)
        
        eval_dataset = pd.DataFrame({
            'item': items, 
            'label': labels, 
            'source': sources, 
            'encoded': encoded_items, 
            'featurized': featurized_items,
            'num_phonemes': num_phonemes,
            'num_features': num_features,
            })
        print("eval dataset:")
        print(eval_dataset.head(5))
        
        print("breakdown of auc test set by num phonemes:")
        for num in eval_dataset['num_phonemes'].unique():
            temp = eval_dataset[eval_dataset['source']==5]
            temp = temp[temp['num_phonemes']==num]
            print(f'{num}: {len(temp)}')
        print("breakdown by test type: ")
        print(eval_dataset['source'].value_counts())
        
    elif eval_humans:
        raise NotImplementedError("please choose a supported combination of features and evaluation settings!")
      
        

    if eval_humans:
        if args.feature_type == 'english':
            forbidden_data_that_cannot_be_queried_about = list(eval_dataset['encoded'].values)
        else:
            forbidden_data_that_cannot_be_queried_about = [item[1] for item in broad_test_set] + [item[1] for item in narrow_test_set] #broad_test_set+narrow_test_set
    else:
        forbidden_data_that_cannot_be_queried_about = []
    linear_train_dataset = dataset.data
    if args.shuffle_train:
        random.shuffle(linear_train_dataset)

    hw_scorer = scorers.HWScorer(dataset, feature_type=args.feature_type)
    eval_informant = informants.HWInformant(dataset, hw_scorer)
    informant = eval_informant

    logs = {}
    unique_features = pd.read_csv('all_features_in_data_unique.csv')['X1'].unique()

    last_costs = None
    last_costs_by_feat = {}

    if eval_humans:
        out_human_evals = get_out_file("HoldoutEvals.csv", args.exp_dir) 
        out_human_evals.write("Step,Run,Strategy,N_INIT,Item,Cost,Source,TestType\n")
        broad_human_evals_writer = get_csv_writer("BroadHoldoutEvals.csv", args.exp_dir) 
        broad_human_evals_writer.writerow(["Step", "Run", "Strategy", "N_INIT", "Items", "Costs", "IsLicit", "IsTI"])
#    eval_metrics = get_out_file("ModelEvalLogs.csv", args.exp_dir)
#    eval_metrics.write("ent,good,bad,diff,acc,rej,Step,Run,Strategy,N_Init,IsTI,judgement,proposed_form,entropy_before,entropy_after,entropy_diff,change_in_probs\n") # including things it queried about


    results_by_observations_writer = get_csv_writer("ResultsByObservations.csv", args.exp_dir)
    results_by_observations_writer.writerow(["Step", "Run", "strategy", "candidate", "judgment", "new_probs", "log_p_all_off", "update_sum"])

    super_strategies = ["eig_train_model", "kl_train_model", "eig_train_history", "kl_train_history", "kl_train_mixed", "eig_train_mixed"]

    for N_INIT in [args.num_init]:
        num_runs = args.num_runs 

        for run in range(args.start_run, args.start_run+num_runs):
            #for strategy in ["train","entropy","unif","max","std","diff"]: # ,"max","unif","interleave","diff","std"
#            for strategy in ["", "eig", "unif","train"]: # only train, entropy, eig, and unif are well-defined here
#            for strategy in ["entropy","entropy_pred","train","unif"]:#,"entropy_pred","train","eig","unif","entropy"]:#"entropy_pred", "entropy","train", "unif","eig",]: # only train, entropy, eig, and unif are well-defined here
#            for strategy in ["kl", "eig", "train", "unif", "entropy", "entropy_pred","kl_train","eig_train"]:

            dir = "/raid/lingo/alexisro/wandb"

#            while True:
#                if not os.access(dir, os.W_OK):
#                    print(f"Lost connection to {dir}")
            
            for strategy in args.strategies: 
                print("STRATEGY:", strategy)

                if args.metric_expect_assume_labels and strategy not in ["eig", "kl", "eig_train_model", "eig_train_mixed", "eig_train_history", "kl_train_model", "kl_train_history", "kl_train_mixed"]:
                    print("args.metric_expect_assume_labels is True for a strategy that doesn't use this argument; ignoring")
                if args.train_expect_type is not None and strategy not in ["eig_train_model", "kl_train_model"]: 
                    print("args.train_expect_type is not None for a strategy that doesn't use this argument; ignoring")

                if args.do_plot_wandb:
                    config = {
                            "n_init": N_INIT, "run": run, "strategy": strategy, 
                            "log_log_alpha_ratio": args.log_log_alpha_ratio, 
                            "prior_prob": args.prior_prob, "feature_type": args.feature_type, 
                            "converge_type": args.converge_type, "tolerance": args.tolerance, 
                            "n_init": N_INIT, "lexicon_file": args.lexicon_file, "warm_start": args.warm_start, 
                            "num_candidates": args.num_candidates,
                            "max_updates_observe": args.max_updates_observe,
                            "max_updates_propose": args.max_updates_propose,
                            "pool_prop_edits": args.pool_prop_edits,
                            "metric_expect_assume_labels": args.metric_expect_assume_labels,
                            "train_expect_type": args.train_expect_type,
                            }
                    tags = [] if args.tags is None else [t.strip() for t in args.tags.split(",")]
                    wandb_run = wandb.init(config=config, project=args.wandb_project, name=strategy, reinit=True, tags = tags, entity="lm-informants") 
                #if strategy == "train":
                #    run = 19
                index_of_next_item = 0
                #print(dataset.data[0])
                random.seed(run)
                np.random.seed(run)
                dataset.random.seed(run)
                linear_train_dataset = dataset.data.copy()
                if args.shuffle_train:
                    random.shuffle(linear_train_dataset) # turn this off if you want to have train be always the same order.
                    
                init_examples = []
                for _ in range(N_INIT):
                    init_examples.append(linear_train_dataset[index_of_next_item])
                    index_of_next_item += 1

                if strategy not in logs:
                    logs[strategy] = []
                log = []
                learner = learners.VBLearner(dataset, strategy=strategy, linear_train_dataset = linear_train_dataset, index_of_next_item = index_of_next_item, seed=run) 
                learner.initialize(n_hyps=1, log_log_alpha_ratio=args.log_log_alpha_ratio, prior_prob=args.prior_prob, converge_type=args.converge_type, feature_type=args.feature_type, tolerance=args.tolerance, warm_start=args.warm_start, features=filtered_features, max_updates_propose=args.max_updates_propose, max_updates_observe=args.max_updates_observe)

                all_features = learner.all_features(return_indices=True)
                unique_feature_indices = [f_idx for (_, f, f_idx) in all_features if f in unique_features]
                
                wandb_table_data = []

                #scores = evaluate_with_external_data(good_dataset,bad_dataset, eval_informant, learner)
#                scores = evaluate(dataset, eval_informant, learner)

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
#                for i in range(args.num_steps-N_INIT):
                encountered_features = set() 
                observed_features = set()
                #UNCOMMENT BELOW TO GET THE FEATURES IN THE TEST SET
                # o = pd.read_csv("WordsAndScores.csv")
                # p = o.loc[o["Source"].isin([1, 2, 3, 4])]["Word"].tolist()   # hithere
                # print(p)
                # z = open("temp_out.tsv","w",encoding="utf8")
                # for word in p:
                #     if word != " ":
                #         print(word)
                #         #assert False
                #         candidate = dataset.vocab.encode([BOUNDARY]+word.split(' ')+[BOUNDARY])
                #         featurized_candidate = set(learner.hypotheses[0]._featurize(candidate).nonzero()[0])
                #         print(word, candidate, featurized_candidate)
                #         z.write(str(word)+'\t'+str(candidate)+"\t"+str(list(featurized_candidate))+"\n")
                #         z.flush()
                # z.close()
                auc_streak = 0
                steps, aucs = [], []

                for i in range(args.num_steps):
                    print("\n\n\n")
                    print(f"i: {i}")
                    step = i
#                    step=N_INIT+i
                    p = learner.hypotheses[0].probs
                   
                    start_time = time.time()
                    #learner.cost()
                    if i < N_INIT:
                        candidate = init_examples[i]
                        judgment = informant.judge(candidate)
                        if args.reverse_judgments:
                            assert isinstance(judgment, bool)
                            judgment = not judgment
                            assert isinstance(judgment, bool)
                        learner.observe(candidate, judgment, update=False)
                        continue
    
                    # some of these args only used by certain strategies (like train_expect_type, metric_expect_assume_labels, informant)
                    candidate = learner.propose(
                            n_candidates=args.num_candidates, 
                            forbidden_data = forbidden_data_that_cannot_be_queried_about, 
                            length_norm=True, 
                            train_expect_type=args.train_expect_type, 
                            verbose=args.verbose, 
                            prop_edits=args.pool_prop_edits, 
                            metric_expect_assume_labels=args.metric_expect_assume_labels, 
                            informant=informant)
                    
                    end_time = time.time()
                    propose_duration = (end_time-start_time)/60

                    str_candidate = str(dataset.vocab.decode(candidate))
                    
                    mean_field_scorer = learner.hypotheses[0]
                    featurized_candidate = mean_field_scorer._featurize(candidate).nonzero()[0]
                    encountered_features.update(set(featurized_candidate))
                    judgment = informant.judge(candidate)
                    if args.reverse_judgments:
                        assert isinstance(judgment, bool)
                        judgment = not judgment
                        assert isinstance(judgment, bool)
                    #prior_probability_of_accept = learner.cost(candidate)


                    #print("ent", (p * np.log(p) + (1-p) * np.log(1-p)).mean())
                    #scores = evaluate_with_external_data(good_dataset,bad_dataset, eval_informant, learner)
#                    scores = evaluate(dataset, eval_informant, learner)
                    #print(dataset.vocab.decode(candidate))


                    total_features = []
                    total_features_2 = []
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
                    if eval_humans and args.feature_type != "english":
                        for item, encoded_word in narrow_test_set:
                            c = learner.cost(encoded_word)
                            #j = informant.cost(encoded_word)
                            #corral_of_judged_human_forms.append((item,c))
                            str_item = " ".join(item)
                            out_human_evals.write(str(step)+','+str(run)+','+str(strategy)+','+str(N_INIT)+","+str_item+","+str(c)+","+str("LEARNER")+",NarrowTest"+'\n')
                            out_human_evals.flush()

                        items, labels, TIs, costs = [], [], [], []

                        for item_idx, (item, encoded_word, featurized) in tqdm(enumerate(broad_test_set)):
                            c = learner.cost(encoded_word, features=featurized)
                            costs.append(c)

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
                    elif eval_humans and args.feature_type == "english":
                        items, costs = [], []
                        labels = list(eval_dataset['label'].values)
                        sources = list(eval_dataset['source'].values)
                        num_phonemes = list(eval_dataset['num_phonemes'].values)
                        num_features = list(eval_dataset['num_features'].values)

                        for item_idx, row in eval_dataset.iterrows():
                            item = row['item']
                            str_item = " ".join(item)
                            items.append(str_item)
                            encoded_word = row['encoded']
                            featurized = row['featurized']
                            label = row['label']
                            source = row['source']
                            
                            c = learner.cost(encoded_word, features=featurized)
                            costs.append(c)

                            str_item = " ".join(item)
                            out_human_evals.write(str(step)+','+str(run)+','+str(strategy)+','+str(N_INIT)+","+str_item+","+str(c)+","+str("LEARNER")+",BroadTest"+'\n')
                            out_human_evals.flush()

                        broad_human_evals_writer.writerow([step, run, strategy, N_INIT, items, costs, labels, sources])
                        print(f"avg cost for broad test set: {np.mean(costs)}")

                    #scores["external_wugs"] = corral_of_judged_human_forms
#                    scores["step"] = N_INIT + i
#                    scores["step"] = i
#                    scores["run"] = run
                    #print(t)
                    #assert False
#                    log.append(scores)
                    print()
#                    print(f"strategy: {strategy}, run: {run}/{num_runs}, step: {N_INIT + i}")
                    print(f"strategy: {strategy}, run: {run}/{num_runs}, step: {i}")
                    
                    #for feat, cost in learner.top_features():
                    #    print(feat, cost)
                    #print()
                    if write_out_feat_probs:
                        for cost, feat in learner.all_features():
                            feat_evals.write(str(N_INIT)+","+str(feat)+','+str(cost)+","+str(step)+","+str(dataset.vocab.decode(candidate)).replace(",","")+","+str(judgment)+","+str(strategy)+","+str(is_ti(str(dataset.vocab.decode(candidate)).replace(",","")))+','+str(run)+ '\n')
                            
                            feat_evals.flush()

                    # "ent,good,bad,diff,acc,rej,Step,Run,Strategy,N_Init\n"
                    
                    entropy_before = entropy(learner.hypotheses[0].probs)
                    # TODO: Unique features only available for atr_harmony
                    if args.feature_type in ["atr_harmony"]:
                        entropy_before_unique = entropy(learner.hypotheses[0].probs[unique_feature_indices])
                    # TODO: for english, compute entropy over seen features (where "seen features" includes features in current candidate, before observing) 
                    elif args.feature_type in ["english"]:
                        entropy_before_unique = entropy(learner.hypotheses[0].probs[list(encountered_features)])
                    else:
                        entropy_before_unique = None
                    probs_before = learner.hypotheses[0].probs.copy()
#                    eig = learner.get_eig(candidate).item()
                    eig = None

                    expected_kl = None 
#                    expected_kl = learner.get_ekl(candidate).item()

                    # TODO: set length_norm to be a variable/parameter, but currently it is True in call to propose() below
                    entropy_of_candidate = learner.hypotheses[0].entropy(candidate, length_norm=True, features=featurized_candidate)
                    pred_prob_pos = np.exp(learner.hypotheses[0].logprob(candidate, True, features=featurized_candidate))
                    chosen_strategy = learner.chosen_strategies[-1] 
                    chosen_cand_type = learner.chosen_cand_types[-1] 
                    if chosen_strategy == 'train':
                        assert judgment

                    if args.do_plot_wandb:
                        log_results = {
                                "step": step, 
                                "entropy_over_features": entropy_before, 
                                "chosen_candidate": str(dataset.vocab.decode(candidate)),
                                "cand_type": chosen_cand_type,
                                "cand_type=edited": int(chosen_cand_type == 'edited'),
                                "cand_type=random": int(chosen_cand_type == 'random'),
                                "eig_of_candidate": eig, 
                                "expected_kl_of_candidate": expected_kl, 
                                "pred_prob_pos": pred_prob_pos, 
                                "strategy_used": chosen_strategy,
                                "strategy_used_is_train": int(chosen_strategy == 'train'),
                                "pred_prob_pos": pred_prob_pos, 
                                "entropy_over_unique_features": entropy_before_unique, 
                                "entropy_of_candidate": entropy_of_candidate, 
                                # this includes in the current candidate, before observing
                                "num_features_encountered": len(encountered_features),
                                # this excludes the current candidate 
                                "num_features_observed": len(observed_features),
                                }
                        if eval_humans:
                            if args.feature_type == "atr_harmony":
                                auc = eval_auc(costs, labels)
                                log_results["auc"] = auc 
                                aucs.append(auc)
                                
                            elif args.feature_type == "english":
                                corrs_df, auc_results, costs_df = eval_corrs(costs, labels, sources, items, num_phonemes, num_features)
                                auc = auc_results['auc']
                                log_results["auc"] = auc
                                aucs.append(auc)
                                c = wandb.Table(dataframe=costs_df)
    
                                table = wandb.Table(dataframe=corrs_df)
                            
                                wandb.log({"corrs": table})
                                wandb.log({"costs": c})
                               
                                corrs_df = corrs_df.set_index('sources')['spearman_corr']
                                log_results.update({f"human_correlation_{k}": v for k, v in corrs_df.to_dict().items()})
                                log_results.update({f"auc_results/{k}": v for k, v in auc_results.items()})

                            else:
                                raise NotImplementedError("Please select a valid feature type!")
                                
#                            if log_results["auc"] >= 0.97:
#                                auc_streak += 1
#                            else:
#                                auc_streak = 0
                    
                    steps.append(step)
                    

                    start_time = time.time()
                    learner.observe(candidate, judgment, verbose=args.verbose, do_plot_wandb=args.do_plot_wandb, batch=args.batch)
                    end_time = time.time()
                    update_duration = (end_time-start_time)/60
                   
                    efficiency_results = {"step": step}
                    efficiency_results["update_duration_mins"] = update_duration
                    efficiency_results["propose_duration_mins"] = propose_duration 
                    
                    if args.do_plot_wandb:
                        wandb.log(log_results)
                        wandb.log({f'efficiency/{k}': v for k,v in efficiency_results.items()})
                    
                    observed_features.update(set(featurized_candidate))

                    if args.do_plot_wandb:
                        all_features = learner.all_features(return_indices=True)
                
                        # for atr_harmony, only look at features in "unique_features"
                        if args.feature_type in ["atr_harmony"]:
                            all_features = [(c, f, f_idx) for (c, f, f_idx) in all_features if f in unique_features]
                        
                        # for english, only look at features that have been seen so far (including in candidate being proposed)
                        elif args.feature_type in ["english"]:
                            all_features = [(c, f, f_idx) for (c, f, f_idx) in all_features if f_idx in encountered_features]

                        all_features.sort(key=lambda x: x[-1])
                        costs = [x[0] for x in all_features]
                        features = [str(x[-1]) for x in all_features]

                        if args.feature_type in ["english"]:
                            # to get the values to determine which points moved, get the costs for features;
                            # if they are not in last_costs_by_feat, they are just the prior prob (TODO: is this robust? it should be because features are only ever added?)
                            last_filtered_costs = [last_costs_by_feat[f_idx] if f_idx in last_costs_by_feat.keys() else args.prior_prob for (_, _, f_idx) in all_features]
                        else:
                            last_filtered_costs = last_costs
                        #
                        # title = f'Step: {step}\nLast candidate (word): {str_candidate}\nFeaturized:{featurized_candidate}\nJudgment:{judgment}'
                        # feature_probs_plot = plot_feature_probs(features, costs, last_filtered_costs, title=title)
                        # last_costs = costs.copy()
                        # last_costs_by_feat = {x[2]: x[0] for x in all_features}
                        #
                        # wandb.log({"feature_probs/plot": wandb.Image(feature_probs_plot)})
                        # plt.close()

                        if i == 0:
                            wandb.log({"features": wandb.Table(columns=["feature_idx", "feature"], data=[[f[-1], f[-2]] for f in all_features])})
                    
                    last_result = learner.results_by_observations[-1] 
                    last_result_DL = {k: [dic[k] for dic in last_result] for k in last_result[0]}
                    results_by_observations_writer.writerow([i, run, strategy, dataset.vocab.decode(candidate), judgment, last_result_DL["new_probs"], last_result_DL["log_p_all_off"], last_result_DL["update_sum"]])
                    
                    probs_after = learner.hypotheses[0].probs.copy()

                    entropy_after = entropy(learner.hypotheses[0].probs)
                    if args.feature_type in ["atr_harmony","english"]:
                        entropy_after_unique = entropy(learner.hypotheses[0].probs[unique_feature_indices])
                    elif args.feature_type in ["english"]:
                        entropy_after_unique = entropy(learner.hypotheses[0].probs[list(encountered_features)])
                    else:
                        entropy_after_unique = None
                    change_in_probs = np.linalg.norm(probs_after-probs_before)
                    print("candidate: ", str_candidate, judgment)
                    print("chosen strategy: ", chosen_strategy)
                    print("chosen candidate type: ", chosen_cand_type)
                    print("entropy before: ", entropy_before) 
                    print("entropy after: ", entropy_after)
                    entropy_diff = entropy_before-entropy_after
                    print("entropy diff: ", entropy_diff)
                    actual_kl = kl_bern(probs_after, probs_before).sum()
                    print("actual kl: ", actual_kl)
                    if args.feature_type in ["atr_harmony", "english"]:
                        entropy_diff_unique = entropy_before_unique-entropy_after_unique
                    else:
                        entropy_diff_unique = None
                    # TODO: add expected information gain

#                    eval_metrics.write(str((p * np.log(p) + (1-p) * np.log(1-p)).mean())+",")
#                    for k, v in scores.items():
#                        #print(f"{k:8s} {v:.4f}")
#                        eval_metrics.write(str(v)+',')
#                    eval_metrics.write(str(step)+','+str(run)+','+str(strategy)+','+str(N_INIT)+","+str(is_ti(str(dataset.vocab.decode(candidate)).replace(",","")))+","+str(judgment)+","+str(dataset.vocab.decode(candidate)).replace(",","")+","+str(entropy_before)+","+str(entropy_after)+","+str(entropy_diff)+","+str(change_in_probs)+'\n')
#                    eval_metrics.flush()
                    wandb_table_data.append([step, strategy, str_candidate, judgment, list(featurized_candidate), eig,  entropy_before, entropy_after, entropy_diff, entropy_before_unique, entropy_after_unique, entropy_diff_unique, change_in_probs, chosen_strategy, chosen_cand_type])


                    results_after_observe = {}
                    results_after_observe['i'] = step
                    results_after_observe['actual_kl'] = actual_kl
                    results_after_observe['actual_entropy_diff'] = entropy_diff 
                    wandb.log({f'results_after_observe/{k}': v for k, v in results_after_observe.items()})
    
#
                    #print("Judging human forms...")
                    #corral_of_judged_human_forms = []

                    # print(learner.hypotheses[0].entropy(candidate, debug=True))

#                    if auc_streak >= 5:
#                        break

                    
                logs[strategy].append(log)
                # create a scatter plot using wandb.plot()

#                data = wandb.Api().run.scan_history(keys=['step', 'auc', 'chosen_candidate'])
#                fig = wandb.plot.scatter(data, x='x', y='y', hover_data=['chosen_candidate'])
#                wandb.log({'interactive_scatter_plot': fig})

#                with open(f"results_{strategy}.json", "w") as writer:
#                    json.dump(logs, writer)
                if args.do_plot_wandb:
                    wandb_table = wandb.Table(columns=["step", "strategy", "proposed_form", "judgment", "features", "eig", "entropy_before", "entropy_after", "entropy_diff", "entropy_before_unique", "entropy_after_unique", "entropy_diff_unique", "change_in_probs", "strategy_for_this_candidate", "cand_type_for_this_candidate"], data=wandb_table_data) 
 
                    wandb.log({"Model Eval Logs": wandb_table})


                # TODO: look at correlations for other domains
                color_map = {"train": "blue", "eig": "orange", "kl": "purple"}
#                if args.feature_type == "atr_harmony" and strategy in super_strategies:
                if strategy in super_strategies:
                    fig = plt.figure()
                    plt.plot(steps, aucs, color="gray")
                    for step, auc, strat in zip(steps, aucs, learner.chosen_strategies):
                        plt.scatter(step, auc, color=color_map[strat])
                    temp_color_map = {k: v for k,v in color_map.items() if k in learner.chosen_strategies}
                    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in temp_color_map.values()]
                    plt.legend(markers, temp_color_map.keys(), numpoints=1)
                    wandb.log({"custom_plots/auc": wandb.Image(fig)})
                    plt.close()

                mean_auc = get_mean_auc(aucs)
                wandb.log({'end_stats/mean_auauc': mean_auc})
                wandb_run.finish()

def get_mean_auc(values, length=None):
    """ if length is not None, extend values to be length long """ 

    if length is not None:
        assert length >= len(values)
        num_to_extend = length - len(values)
        values = values + [values[-1]]*num_to_extend 
    area = 0
    for i in range(len(values) - 1):
        # TODO: is abs() here robust?
        area += (values[i] + values[i + 1]) / 2.0 * 1
    mean_area = area / len(values)
    return mean_area

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--log_log_alpha_ratio", type=float, default=2)
    parser.add_argument("--prior_prob", type=float, default=0.1)
    parser.add_argument("--feature_type", type=str, default="atr_harmony")
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--converge_type', type=str, default="symmetric")
#    parser.add_argument('--tolerance', type=float, default=0.001)
    parser.add_argument('--tolerance', type=float, default=0.001/512)
    parser.add_argument('--num_steps', type=int, default=150)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--tags', type=str, default=None)

    # If you want to give a lexicon file for the train strategy different from the default
    parser.add_argument('--lexicon_file', type=str, default=None)

    parser.set_defaults(verbose=False)
   
    # cold or warm start; defaults to cold (i.e. warm_start=False)
    parser.add_argument('--warm_start', action='store_true', dest='warm_start')
    parser.set_defaults(warm_start=False)
    
    # batch defaults to True
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--no-batch', dest='batch', action='store_false')
    parser.set_defaults(batch=True)
    
    parser.add_argument('--eval_humans', action='store_true')
    parser.add_argument('--no-eval_humans', dest='eval_humans', action='store_false')
    parser.set_defaults(eval_humans=True)
    
    parser.add_argument('--shuffle_train', action='store_true')
    parser.add_argument('--no-shuffle_train', dest='shuffle_train', action='store_false')
    parser.set_defaults(shuffle_train=True)
    
    parser.add_argument('--reverse_judgments', action='store_true')
    parser.set_defaults(reverse_judgments=False)

    parser.add_argument('--profile_name', default='my_profile')

    parser.add_argument('--num_candidates', default=100, type=int)
    parser.add_argument('--start_run', default=0, type=int, help='What run to start from')
    
    parser.add_argument('--min_length', default=2, type=int, help='min length for sampling random sequences')
    parser.add_argument('--max_length', default=5, type=int, help='max length for sampling random sequences')
    
    def parse_max_updates(value):
        if value == 'None':
            return None
        return int(value)

    def parse_proportion(value):
        value = float(value)
        assert value <= 1.0
        assert value >= 0.0
        return value

    parser.add_argument('--max_updates_propose', default=None, type=parse_max_updates, help='max # updates for proposal (only used for eig/ekl)')
    parser.add_argument('--max_updates_observe', default=None, type=parse_max_updates, help='max # updates for observing')
    
    parser.add_argument('--num_init', default=0, type=int) 
    
    parser.add_argument('--train_expect_type', 
            default='proposal_samples', 
            choices=[
                None,
                'proposal_samples', 
                'lexicon_samples', 
                'true_candidate'],
            help=('how to compute the train expectation in'
            'kl_train_model/eig_train_model '
            '(ignored by other strategies); '
            'proposal_samples = using samples from proposal distr; '
            'lexicon_samples = using samples from the lexicon;' 
            'true_candidate = using actual candidate'))
   
    # defaults to false
    parser.add_argument('--metric_expect_assume_labels', action='store_true',
        help=('whether to assume known labels in computing the'
            'expectation in kl_train_model/eig_train_model '
            '(ignored by other strategies)'))
    parser.set_defaults(metric_expect_assume_labels=False)

    strategies = [
            "entropy", "entropy_pred", "unif", "train",
            "eig", "kl", 
            "kl_train_model",
            "eig_train_model",
            "eig_train_history", "eig_train_mixed",
            "kl_train_mixed", "kl_train_history",
            ]
    parser.add_argument('--strategies', nargs='+', required=False, default=strategies)
    parser.add_argument('--pool_prop_edits', default=0.0, type=parse_proportion, help='proportion of proposal pool consisting of edited candidates from lexicon')


    args = parser.parse_args()

    if args.reverse_judgments and args.metric_expect_assume_labels:
        raise NotImplementedError("Not yet implemented to reverse judgments when assuming labels for metric expectation in forward-looking strategies; need to reverse labels there too")

    if args.wandb_project is not None:
        args.do_plot_wandb = True
    else:
        args.do_plot_wandb=False

    print("args: ", args)
    print("strategies: ", args.strategies)

#    os.makedirs('profiles', exist_ok=True)
#    profile_path = f'profiles/{args.profile_name}.prof'
    main(args)
#    cProfile.run('main(args)', filename=profile_path)
#    print("Wrote profiler output to:", profile_path)
#    
