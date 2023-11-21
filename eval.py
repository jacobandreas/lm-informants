import numpy as np
import pandas as pd
from sklearn import metrics
import random
from BayesianGLM import BayesianLearner
from learners import VBLearner
from itertools import product
from tqdm import tqdm
from informants import HWInformant, TrigramInformant
from scorers import MeanFieldScorer, HWScorer
from os.path import exists
from main import read_in_blicks, BOUNDARY
import matplotlib.pyplot as plt
import datasets

def get_train_data(args):
    lexicon = args.lexicon_file or f"data/hw/{args.feature_type}_lexicon.txt"
    assert exists(lexicon) 
    return datasets.load_lexicon(lexicon, min_length=args.min_length, max_length=args.max_length)

def get_test_data(args, train_dataset):
    if args.feature_type == "english":
        df = pd.read_csv('WordsAndScoresFixed_newest.csv')
        items = df['Word'].map(lambda s: s.strip().split(' ')).tolist()
        sources = df['Source'].astype(int).tolist()
        labels = df['Score'].tolist()
        encoded = []
        assert len(items) == len(sources)
        for i in range(len(items)):
            phonemes = [BOUNDARY] + items[i]
            if sources[i] != 1:
                phonemes += [BOUNDARY]
            encoded.append(train_dataset.vocab.encode(phonemes)) 
        return pd.DataFrame({"encoded": encoded, "label": labels})

    if args.feature_type == "atr_harmony":
        items = read_in_blicks("test_set.csv")
    elif args.feature_type == "atr_four":
        items = read_in_blicks(f"atr_four_test_set.txt")
    else:
        raise NotImplementedError(f"No dataset defined for feature_type {args.feature_type}")

    encoded = [train_dataset.vocab.encode([BOUNDARY] + item + [BOUNDARY]) for item in items]
    hw_scorer = HWScorer(
        train_dataset, 
        feature_type=args.feature_type, 
        phoneme_feature_file=args.phoneme_feature_path,
    )
    informant = HWInformant(train_dataset, hw_scorer)
    labels = [informant.judge(e) for e in encoded]
    return pd.DataFrame({"encoded": encoded, "label": labels})

def eval_corrs():
    # for english
    # note eval humans is fake
    pass

def get_auc(learner, dataset):
    probs = np.vectorize(learner.probs)(dataset["encoded"].values)
    labels = dataset['label'].values.astype(int)
    return auc(labels, probs)

def auc(labels, probs):
    fpr, tpr, _ = metrics.roc_curve(labels, probs, pos_label=1)
    return metrics.auc(fpr, tpr)

def init_learner(args, dataset, strategy, seed):
    linear_train_dataset = dataset.data.copy()
    if args.shuffle_train:
        random.shuffle(linear_train_dataset)
    learner = BayesianLearner(
        dataset, 
        seed=seed,
        strategy=strategy, 
        linear_train_dataset=linear_train_dataset,
        index_of_next_item=0,
        feature_type=args.feature_type, 
        track_params=True
    )
    learner.initialize()
    return learner

def train_and_eval_learner(args, learner, informant):
    test_data = get_test_data(args, learner.dataset)
    aucs = [get_auc(learner, test_data)]
    for _ in range(args.n_steps):
        candidate = learner.propose(args.n_candidates)
        judgment = informant.judge(candidate)
        learner.observe(candidate, judgment)
        aucs.append(get_auc(learner, test_data))
    trackers = learner.get_param_trackers()
    trackers["auc"] = aucs
    trackers["step"] = np.arange(args.n_steps+1)
    return trackers

def write_trackers(csv_name, trackers):
    if exists(csv_name):
        keys = set(pd.read_csv(csv_name, index_col=False, nrows=0).columns)
        assert keys == trackers.keys()
        pd.DataFrame.from_dict(trackers).to_csv(csv_name, mode="a", index=False, header=False)
    else:
        pd.DataFrame.from_dict(trackers).to_csv(csv_name, index=False)

def evaluate(args):
    train_data = get_train_data(args)
    if args.use_trigram_oracle:
        assert args.feature_type == "atr_harmony", "Trigram oracle only implemented for atr_harmony"
        mean_field_scorer = MeanFieldScorer(train_data, feature_type=args.feature_type)
        informant = TrigramInformant(train_data, mean_field_scorer, args.trigram_oracle_seed, num_bad_features=16)
    else:
        hw_scorer = HWScorer(train_data, feature_type=args.feature_type)
        informant = HWInformant(train_data, hw_scorer)
    
    for seed, strategy in tqdm(
        product(range(args.n_seeds), args.strategies),
        total=args.n_seeds * len(args.strategies)
    ):
        random.seed(seed)
        np.random.seed(seed)
        train_data.random.seed(seed)

        learner = init_learner(args, train_data, strategy, seed)
        trackers = train_and_eval_learner(args, learner, informant)
        write_trackers(args.csv_name, trackers)
    plot(args.csv_name)

def plot(csv_name, cmap="tab20"):
    data = pd.read_csv(csv_name, index=False)
    strategies = data["strategy"].unique()
    n_seeds = len(data["seed"].unique())

    grouped_data = data.groupby(["strategy", "step"])["auc"]
    means = grouped_data.mean()
    stds = grouped_data.std()

    plt.style.use("ggplot")
    cmap = plt.get_cmap(cmap)
    for i, strategy in enumerate(strategies):
        color = cmap(i/(len(strategies)-1))
        m = means[(strategy,)].values
        ci = 1.96 * stds[(strategy,)].values / np.sqrt(n_seeds)
        plt.plot(range(len(m)), m, label=strategy, color=color)
        plt.fill_between(range(len(m)), m-ci, m+ci, alpha=0.2, color=color)
    plt.xlabel("step")
    plt.ylabel("auc")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(csv_name.replace(".csv", ".pdf"), bbox_inches="tight")






# HYPERPARAMETERS
"""
-include length norm!! ()

-lexicon file is train

-tags and wandb

-don't worry abt n_init
-no reverse judgments etc

-don't use narrow test set
-ignore TI

feature types are ["atr_harmony", "atr_lg_", "english"]

-include trigram informant

-don't need total_features etc

logging:
-mean auc
-strategy used / is train
-params at every step !!! (np.save)
-DO NOT NEED all features log
-maybe track entropy at each step (for seen features vs all?)
"""