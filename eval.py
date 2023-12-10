import numpy as np
import pandas as pd
from sklearn import metrics
import random
from BayesianGLM import BayesianLearner
from tqdm import tqdm
from informants import HWInformant, TrigramInformant
from scorers import MeanFieldScorer, HWScorer
import os
import pathlib
from main import read_in_blicks, BOUNDARY, eval_corrs
import matplotlib.pyplot as plt
import seaborn as sns
import datasets
import argparse
import wandb
from itertools import product
import time
import gc


def evaluate(args):
    train_data = get_train_data(args)
    informant = get_informant(args, train_data)
    for seed in range(args.n_seeds):
        for strategy in args.strategies:
            group = group_by(args, strategy=strategy, seed=seed)
            config = vars(args)
            config["seed"] = seed
            config["strategy"] = strategy
            config["group"] = group
            wandb.init(
                project=args.name,
                entity=args.wandb_team,
                group=group,
                name=f"{strategy}-{seed}",
                mode=args.wandb_mode,
                config=config,
                reinit=True,
                force=True,
            )
            random.seed(seed)
            np.random.seed(seed)
            train_data.random.seed(seed)
            learner = get_learner(args, train_data, strategy, seed)
            trackers = train_and_eval_learner(args, learner, informant)
            write_trackers(args, trackers)
            del learner
            del trackers
            gc.collect()
    make_heatmaps(args)


def group_by(args, **kwargs):
    d = vars(args)
    params = []
    for k in args.group_by:
        params.append(
            f"{k}={kwargs[k]}"
            if k in kwargs
            else f"{k}={d[k]}"
        )
    group = "_".join(params)
    return group


def train_and_eval_learner(args, learner, informant):
    test_data = get_test_data(args, learner.dataset)
    forbidden_seqs = test_data["encoded"].values.tolist()
    init_results = eval_learner(learner, test_data)
    all_results = [init_results]
    times = [0]
    log_step(0, 0, learner, init_results)
    write_params(args, learner, 0)
    for step in tqdm(
        range(1, args.n_steps + 1),
        unit="step",
        desc=f"{learner.strategy}-{learner.seed}",
    ):
        t0 = time.time()
        candidate = learner.propose(args.n_candidates, forbidden_seqs)
        judgment = informant.judge(candidate)
        learner.observe(candidate, judgment)
        time_elapsed = time.time() - t0
        
        results = eval_learner(learner, test_data)
        all_results.append(results)
        times.append(time_elapsed)
        log_step(step, time_elapsed, learner, results)
        write_params(args, learner, step)
        gc.collect()
    trackers = learner.get_param_trackers()
    trackers["time"] = times
    trackers["step"] = np.arange(args.n_steps + 1)
    trackers["seed"] = [learner.seed]*(args.n_steps + 1)
    trackers["strategy"] = [learner.strategy]*(args.n_steps + 1)
    for k in all_results[0]:
        trackers[k] = [r[k] for r in all_results]
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


def eval_learner(learner, dataset):
    if learner.feature_type == "english":
        return eval_learner_english(learner, dataset)
    return eval_learner_general(learner, dataset)


def eval_learner_general(learner, dataset):
    labels = dataset["label"].values.astype(int)
    probs = np.array([learner.probs(encoded) for encoded in dataset["encoded"]])
    logits = np.array([learner.logits(encoded) for encoded in dataset["encoded"]])
    results = {}
    results["auc"] = auc(labels, probs)
    results["log-likelihood"] = log_likelihood(labels, probs, logits)
    results["accuracy"] = metrics.accuracy_score(labels, probs>0.5)
    results["f1"] = metrics.f1_score(labels, probs>0.5)
    return results


def eval_learner_english(learner, dataset):
    corrs_df, auc_results, costs_df = eval_corrs(
        [learner.cost(encoded).item() for encoded in dataset["encoded"]],
        dataset["label"],
        dataset["source"],
        dataset["item"],
        dataset["num_phonemes"],
        dataset["num_features"],
    )
    results = {
        "costs": wandb.Table(dataframe=costs_df),
        "corrs": wandb.Table(dataframe=corrs_df),
    }
    results.update(
        {f"human_correlation_{k}": v for k, v in corrs_df.to_dict().items()}
    )
    results.update({f"auc_results/{k}": v for k, v in auc_results.items()})
    results.update(eval_learner_general(learner, dataset[dataset["source"]==5]))
    return results


def auc(labels, probs):
    fpr, tpr, _ = metrics.roc_curve(labels, probs, pos_label=1)
    return metrics.auc(fpr, tpr)


def log_likelihood(labels, probs, logits):
    with np.errstate(divide = 'ignore'):
        true_logprobs = np.log(probs)
        false_logprobs = np.log(1-probs)
    true_logprobs[probs==0] = logits[probs==0]
    false_logprobs[probs==1] = -logits[probs==1]
    assert np.all(-np.inf < true_logprobs) and np.all(true_logprobs < np.inf)
    assert np.all(-np.inf < false_logprobs) and np.all(false_logprobs < np.inf)
    return np.sum(true_logprobs[labels==1]) + np.sum(false_logprobs[labels==0])


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
        phoneme_feature_file="data/hw/atr_harmony_features.txt"
        if args.feature_type.startswith("atr_lg_")
        else None,
        # scorer hyperparams
        alpha_prior_mu=args.alpha_prior_mu,
        alpha_prior_sigma=args.alpha_prior_sigma,
        beta_prior_mu=args.beta_prior_mu,
        beta_prior_sigma=args.beta_prior_sigma,
        step_size=args.step_size,
        n_updates=args.n_updates,
        use_mean=args.use_mean,
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
    trackers.update({k: [v]*(args.n_steps + 1) for k,v in vars(args).items()})
    path = make_folders_for_path(f"{args.output_dir}/{args.csv_name}")
    if os.path.exists(path):
        keys = set(pd.read_csv(path, index_col=False, nrows=0).columns)
        assert keys == trackers.keys(), f"mismatch: {keys.symmetric_difference(trackers.keys())}"
        pd.DataFrame.from_dict(trackers).to_csv(
            path, mode="a", index=False, header=False
        )
    else:
        pd.DataFrame.from_dict(trackers).to_csv(path, index=False)


def write_params(args, learner, step):
    p = learner.hypothesis.params
    for dist, param in product(["alpha", "beta"], ["mu", "sigma"]):
        path = make_folders_for_path(
            f"{args.output_dir}/seed-{learner.seed}/{learner.strategy}/{dist}/{param}/step-{step}"
        )
        np.save(path, p[f"{dist}_posterior_{param}"])


def log_step(step, time_elapsed, learner, results):
    log = {
        "step": step,
        "time": time_elapsed,
        "last_proposed_is_train": int(learner.last_proposed == "train"),
        "n_observed_train": learner.n_observed_train,
        "n_observed_metric": learner.n_observed_metric,
        "train_avg": learner.train_avg,
        "metric_avg": learner.metric_avg,
        "alpha_mu": learner.alpha_mu[-1],
        "alpha_sigma": learner.alpha_sigma[-1],
        "avg_beta_mu": learner.avg_beta_mu[-1],
        "avg_beta_sigma": learner.avg_beta_sigma[-1],
        "avg_seen_beta_mu": learner.avg_seen_beta_mu[-1],
        "avg_seen_beta_sigma": learner.avg_seen_beta_sigma[-1],
        "avg_unseen_beta_mu": learner.avg_unseen_beta_mu[-1],
        "avg_unseen_beta_sigma": learner.avg_unseen_beta_sigma[-1],
        "pct_good_examples": learner.pct_good_examples[-1],
    }
    log.update(results)
    log.update({f"{learner.strategy}/{k}":v for k,v in log.items()})
    wandb.log(log)


def make_folders_for_path(path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_heatmaps(args):
    output_path = os.path.join(args.output_dir, "heatmaps")
    csv_path = os.path.join(args.output_dir, args.csv_name)
    data = pd.read_csv(csv_path, index_col=False)
    strategies = data["strategy"].unique().tolist()
    final_step = data[data["step"]==np.max(data["step"])]
    for metric in ["auc", "f1", "accuracy", "log-likelihood"]:
        low = np.min(final_step[metric])
        high = np.max(final_step[metric])
        for strategy in strategies:
            plt.clf()
            d = final_step[final_step["strategy"]==strategy]
            m = d.groupby(args.group_by)[metric].mean().unstack(level=0)
            sns.heatmap(m, annot=True, fmt=".4g", vmin=low, vmax=high)
            plt.title(f"{strategy} {metric}")
            save_path = make_folders_for_path(os.path.join(output_path, f"{strategy}_{metric}.pdf"))
            plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--wandb_team", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--lexicon_file", type=str, default=None)
    parser.add_argument("--feature_type", type=str, default="atr_harmony")
    parser.add_argument("--min_length", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=5)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--n_steps", type=int, default=150)
    parser.add_argument("--n_candidates", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--csv_name", type=str, default="results.csv")
    parser.add_argument("--output_dir", type=str, default=f"outputs")
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
    parser.add_argument("--group_by", nargs="+", default=["seed"])
    # scorer hyperparams
    parser.add_argument("--alpha_prior_mu", type=float, default=5.0)
    parser.add_argument("--alpha_prior_sigma", type=float, default=1.0)
    parser.add_argument("--beta_prior_mu", type=float, default=-10.0)
    parser.add_argument("--beta_prior_sigma", type=float, default=20.0)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--n_updates", type=int, default=2000)
    parser.add_argument("--use_mean", action="store_true")
    args = parser.parse_args()
    evaluate(args)