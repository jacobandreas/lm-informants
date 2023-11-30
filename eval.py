import numpy as np
import pandas as pd
from sklearn import metrics
import random
from BayesianGLM import BayesianLearner
from tqdm import tqdm
from informants import HWInformant, TrigramInformant
from scorers import MeanFieldScorer, HWScorer
import os
from main import read_in_blicks, BOUNDARY, eval_corrs
import matplotlib.pyplot as plt
import datasets
import argparse
import wandb
from functools import reduce
from itertools import product


def evaluate(args):
    train_data = get_train_data(args)
    informant = get_informant(args, train_data)
    for seed in range(args.n_seeds):
        wandb.init(
            project=args.run_name,
            name=f"seed={seed}",
            config=vars(args),
            reinit=True,
            force=True,
        )
        for strategy in args.strategies:
            random.seed(seed)
            np.random.seed(seed)
            train_data.random.seed(seed)
            learner = get_learner(args, train_data, strategy, seed)
            trackers = train_and_eval_learner(args, learner, informant)
            write_trackers(args, trackers)
    path = os.path.join(args.output_dir, args.csv_name)
    plot(path)


def train_and_eval_learner(args, learner, informant):
    test_data = get_test_data(args, learner.dataset)
    forbidden_seqs = test_data["encoded"].values.tolist()
    init_auc = get_auc(learner, test_data)
    aucs = [init_auc]
    log_step(0, init_auc, learner)
    write_params(args, learner, 0)
    for step in tqdm(
        range(1, args.n_steps + 1),
        unit="step",
        desc=f"{learner.strategy}-{learner.seed}",
    ):
        candidate = learner.propose(args.n_candidates, forbidden_seqs)
        judgment = informant.judge(candidate)
        learner.observe(candidate, judgment)
        if args.feature_type == "english":
            auc, additional_logs = eval_english(learner, test_data)
        else:
            auc, additional_logs = get_auc(learner, test_data), None
        aucs.append(auc)
        log_step(step, auc, learner, additional_logs)
        write_params(args, learner, step)
    trackers = learner.get_param_trackers()
    trackers["seed"] = learner.seed
    trackers["strategy"] = learner.strategy
    trackers["auc"] = aucs
    trackers["step"] = np.arange(args.n_steps + 1)
    return trackers


def get_train_data(args):
    if args.lexicon_file:
        lexicon = args.lexicon_file
    else:
        if args.feature_type.startswith("atr_lg_"):
            lexicon = f"data/generated_langs/{args.feature_type}_lexicon.txt"
        else:
            lexicon = f"data/hw/{args.feature_type}_lexicon.txt"
    assert os.path.exists(lexicon)
    return datasets.load_lexicon(
        lexicon, min_length=args.min_length, max_length=args.max_length
    )


def get_test_data(args, train_dataset):
    if args.feature_type == "english":
        df = pd.read_csv("WordsAndScoresFixed_newest.csv")
        items = df["Word"].map(lambda s: s.strip().split(" ")).tolist()
        sources = df["Source"].astype(int).tolist()
        labels = df["Score"].tolist()
        encoded = []
        num_phonemes = []
        num_features = []
        mean_field_scorer = MeanFieldScorer(train_dataset, feature_type="english")
        assert len(items) == len(sources)
        for i in range(len(items)):
            num_phonemes.append(len(items[i]))
            phonemes = [BOUNDARY] + items[i]
            if sources[i] != 1:
                phonemes += [BOUNDARY]
            encoding = train_dataset.vocab.encode(phonemes)
            encoded.append(encoding)
            num_features.append(
                len(mean_field_scorer._featurize(encoding).nonzero()[0])
            )
        return pd.DataFrame(
            {
                "encoded": encoded,
                "label": labels,
                "num_phonemes": num_phonemes,
                "num_features": num_features,
                "item": items,
                "source": sources,
            }
        )

    if args.feature_type.startswith("atr_lg_"):
        data = pd.read_csv(f"data/generated_langs/{args.feature_type}_test_set.csv")
        encoded = [
            train_dataset.vocab.encode([BOUNDARY] + item.split(" ") + [BOUNDARY])
            for item in data["item"]
        ]
        return pd.DataFrame({"encoded": encoded, "label": data["label"].tolist()})

    if args.feature_type == "atr_harmony":
        items = read_in_blicks("test_set.csv")
    elif args.feature_type == "atr_four":
        items = read_in_blicks(f"atr_four_test_set.txt")
    else:
        raise NotImplementedError(
            f"No dataset defined for feature_type {args.feature_type}"
        )
    encoded = [
        train_dataset.vocab.encode([BOUNDARY] + item + [BOUNDARY]) for item in items
    ]
    hw_scorer = HWScorer(
        train_dataset,
        feature_type=args.feature_type,
    )
    informant = HWInformant(train_dataset, hw_scorer)
    labels = [informant.judge(e) for e in encoded]
    return pd.DataFrame({"encoded": encoded, "label": labels})


def get_auc(learner, dataset):
    probs = [learner.probs(encoded) for encoded in dataset["encoded"]]
    labels = dataset["label"].values.astype(int)
    return auc(labels, probs)


def auc(labels, probs):
    fpr, tpr, _ = metrics.roc_curve(labels, probs, pos_label=1)
    return metrics.auc(fpr, tpr)


def eval_english(learner, dataset):
    corrs_df, auc_results, costs_df = eval_corrs(
        [learner.cost(encoded).item() for encoded in dataset["encoded"]],
        dataset["label"],
        dataset["source"],
        dataset["item"],
        dataset["num_phonemes"],
        dataset["num_features"],
    )
    auc = auc_results["auc"]
    additional_logs = {
        "costs": wandb.Table(dataframe=costs_df),
        "corrs": wandb.Table(dataframe=corrs_df),
    }
    additional_logs.update(
        {f"human_correlation_{k}": v for k, v in corrs_df.to_dict().items()}
    )
    additional_logs.update({f"auc_results/{k}": v for k, v in auc_results.items()})
    return auc, additional_logs


def get_learner(args, dataset, strategy, seed):
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
        track_params=True,
        use_mean=args.use_mean,
        phoneme_feature_file="data/hw/atr_harmony_features.txt"
        if args.feature_type.startswith("atr_lg_")
        else None,
    )
    learner.initialize()
    return learner


def get_informant(args, train_dataset):
    if args.feature_type.startswith("atr_lg_"):
        with open(
            f"data/generated_langs/{args.feature_type}_trigram_features.txt", "r"
        ) as f:
            bad_features = [int(l) for l in f.readlines()]
        mean_field_scorer = MeanFieldScorer(
            train_dataset,
            feature_type=args.feature_type,
            phoneme_feature_file="data/hw/atr_harmony_features.txt",
        )
        return TrigramInformant(train_dataset, mean_field_scorer, bad_features)
    else:
        hw_scorer = HWScorer(train_dataset, feature_type=args.feature_type)
        return HWInformant(train_dataset, hw_scorer)


def write_trackers(args, trackers):
    path = make_folders_for_path(f"{args.output_dir}/{args.csv_name}")
    if os.path.exists(path):
        keys = set(pd.read_csv(path, index_col=False, nrows=0).columns)
        assert keys == trackers.keys()
        pd.DataFrame.from_dict(trackers).to_csv(
            path, mode="a", index=False, header=False
        )
    else:
        pd.DataFrame.from_dict(trackers).to_csv(path, index=False)


def write_params(args, learner, step):
    p = learner.hypothesis.params
    for dist, param in product(["alpha", "beta"], ["mu", "sigma"]):
        path = make_folders_for_path(
            f"{args.output_dir}/{learner.strategy}-{learner.seed}/{dist}/{param}/{step}"
        )
        np.save(path, p[f"{dist}_posterior_{param}"])


def log_step(step, auc, learner, additional_logs=None):
    strategy = learner.strategy
    log = {
        f"{strategy}/last_proposed_is_train": int(learner.last_proposed == "train"),
        f"{strategy}/n_observed_train": learner.n_observed_train,
        f"{strategy}/n_observed_metric": learner.n_observed_metric,
        f"{strategy}/train_avg": learner.train_avg,
        f"{strategy}/metric_avg": learner.metric_avg,
        f"{strategy}/alpha_mu": learner.alpha_mu[-1],
        f"{strategy}/alpha_sigma": learner.alpha_sigma[-1],
        f"{strategy}/avg_beta_mu": learner.avg_beta_mu[-1],
        f"{strategy}/avg_beta_sigma": learner.avg_beta_sigma[-1],
        f"{strategy}/avg_seen_beta_mu": learner.avg_seen_beta_mu[-1],
        f"{strategy}/avg_seen_beta_sigma": learner.avg_seen_beta_sigma[-1],
        f"{strategy}/avg_unseen_beta_mu": learner.avg_unseen_beta_mu[-1],
        f"{strategy}/avg_unseen_beta_sigma": learner.avg_unseen_beta_sigma[-1],
        f"{strategy}/pct_good_examples": learner.pct_good_examples[-1],
        f"{strategy}/auc": auc,
        f"step": step,
    }
    if additional_logs:
        log.update({f"{strategy}/{k}": v for k, v in additional_logs.items()})
    wandb.log(log)


def make_folders_for_path(path):
    def f(path, sub_folder):
        new_path = os.path.join(path, sub_folder)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        return new_path

    reduce(f, path.split("/")[:-1])
    return path


def plot(csv_name, cmap="tab20"):
    data = pd.read_csv(csv_name, index_col=False)
    strategies = data["strategy"].unique().tolist()
    n_seeds = len(data["seed"].unique())

    grouped_data = data.groupby(["strategy", "step"])["auc"]
    means = grouped_data.mean()
    stds = grouped_data.std()

    plt.style.use("ggplot")
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, num=len(strategies))]
    strategies.sort(key=lambda s: -means[(s,)].iloc[-1])
    for i, strategy in enumerate(strategies):
        m = means[(strategy,)].values
        ci = 1.96 * stds[(strategy,)].values / np.sqrt(n_seeds)
        plt.plot(
            range(len(m)),
            m,
            label=f"{strategy} (auc={'{:.2f}'.format(m[-1])})",
            color=colors[i],
        )
        plt.fill_between(range(len(m)), m - ci, m + ci, alpha=0.2, color=colors[i])
    plt.xlabel("step")
    plt.ylabel("auc")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.savefig(csv_name.replace(".csv", ".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="lm-informants")
    parser.add_argument("--lexicon_file", type=str, default=None)
    parser.add_argument("--feature_type", type=str, default="atr_harmony")
    parser.add_argument("--min_length", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=5)
    parser.add_argument("--shuffle_train", type=bool, default=True)
    parser.add_argument("--n_steps", type=int, default=150)
    parser.add_argument("--n_candidates", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--use_mean", type=bool, default=False)
    parser.add_argument("--csv_name", type=str, default="results.csv")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    strategies = [
        "train",
        "unif",
        "entropy",
        "entropy_pred",
        "eig",
        "eig_train_mixed",
        "eig_train_model",
        "eig_train_history",
        "kl",
        "kl_train_mixed",
        "kl_train_model",
        "kl_train_history",
    ]
    parser.add_argument("--strategies", nargs="+", default=strategies)
    args = parser.parse_args()
    evaluate(args)
