import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from main import get_mean_auauc

NUM_SEEDS = 9
STRICT = False


def parse_args():
    # add argument for the path to the dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to results to analyze",
        default="1106_all_atr_generated_1214_full.csv",
    )

    parser.add_argument(
        "--results_path",
        type=str,
        help="path to save results",
        default="cross_validation_results.csv",
    )

    parser.add_argument(
        "--strategy_key",
        default="strategy",
        help="key for strategy column",
        type=str,
    )

    parser.add_argument(
        "--hyperparams",
        default=[
            "prior_prob",
            "max_updates_propose",
            "log_log_alpha_ratio",
        ],
        help="hyperparams to analyze",
        type=str,
        nargs="+",
    )

    parser.add_argument(
        "--auc_metric",
        default="end_results/end_stats/mean_auauc",
        help="auc_metric to analyze",
        type=str,
    )

    parser.add_argument(
        "--data_type",
        default="full",
        choices=["full", "summary"],
        help="type of data to analyze",
        type=str,
    )

    return parser.parse_args()


def subset_results(results, config, strategy):
    """Subsets results according to config and strategy.
    Args:
        config (dict): config to subset on
            for each key, value pair in config, subset results to those where column f'config/{key}' == value
        strategy (str): strategy to subset on (ie where column 'strategy' == strategy)
    """
    for key, value in config.items():
        results = results[results[f"config/{key}"] == value]
    results = results[results[args.strategy_key] == strategy]
    return results


def validate_auauc(results, args):
    # breakpoint()
    print("Validating auauc...")
    # Sanity check that the mean_auauc are the same as if we re-computed it
    # Sample a row
    row = results.sample()
    # Get hyperparams associated with this row
    row_config = {
        key: row[f"config/{key}"].values[0] for key in args.hyperparams + ["run"]
    }
    # Get auauc for this row
    row_auauc = row[args.auc_metric].values[0]

    # Subset based on row's config
    results_subset = subset_results(
        results, row_config, row[args.strategy_key].values[0]
    )
    last_step = max(results_subset["step"].unique())
    aucs = list(results_subset["auc"].values)
    # breakpoint()
    assert len(aucs) == last_step + 1, f"aucs: {aucs}, last_step: {last_step}"
    mean_auauc = get_mean_auauc(aucs, length=last_step + 1)
    assert np.isclose(mean_auauc, row_auauc)
    print("Done validating auauc.")


def load_data(args):
    if args.data_type == "full":
        return load_data_full(args)
    elif args.data_type == "summary":
        return load_data_summary(args)
    else:
        raise ValueError(f"Invalid data_type: {args.data_type}")


def load_data_full(args):
    """
    Load data from args.data_path.
    """
    print(f"Loading data from: {args.data_path}...")

    results = pd.read_csv(args.data_path, index_col=0)
    results = results.convert_dtypes()
    # print(results[results.columns[19]].unique())

    # Use dtypes to infer dataptypes, but also allow for nans
    # results = pd.read_csv(args.data_path, dtype=dtypes, na_values=["nan"])
    last_step = max(results["step"].unique())

    # Sanity check that the mean_auauc are the same as if we re-computed it
    # If so, we can just look at the last step
    validate_auauc(results, args)

    # Hacky, but just get last step
    results = results[results["step"] == last_step]

    # Create identifier based on hyperparams
    results["hyperparam_identifier"] = results.apply(get_hyperparam_identifier, axis=1)

    print("Done.")
    return results


def load_data_summary(args):
    """
    Load data from args.data_path.
    """
    print(f"Loading data from: {args.data_path}...")

    results = pd.read_csv(args.data_path, index_col=0)
    results = results.convert_dtypes()

    # Create identifier based on hyperparams
    results["hyperparam_identifier"] = results.apply(get_hyperparam_identifier, axis=1)

    print("Done.")
    return results


def get_cross_validation_results(results):
    """
    Pseudocode for getting results

    test_results_by_strat = {strat: [] for strat in STRATEGIES}
    for test_seed in SEEDS:
        train_results = all results except those with test_seed
        test_results = results for test_seed
        for strategy in STRATEGIES:
            strategy_train_results = subset of train_results for strategy
            strategy_test_results = subset of test_results for strategy
            best_hyperparams = get best hyperparams in strategy_train_results according to median auauc (across seeds)
            test_results = auauc for strategy in test_results using best hyperparams
            test_results_by_strat[strategy].append(test_results)

    # now test_results_by_strat has results for each strat on all folds
    # get median across folds
    median_test_results_by_strat = {strat: median(test_results_by_strat[strat]) for strat in STRATEGIES}
    """

    print("Getting cross validation results...")

    # Get results for each strategy
    all_test_results = []
    test_auaucs_by_strat = {strat: [] for strat in results[args.strategy_key].unique()}

    for test_seed in tqdm(results["config/run"].unique()):
        train_results = results[results["config/run"] != test_seed]
        test_results = results[results["config/run"] == test_seed]

        if len(test_results) == 0:
            print(
                f"test_seed: {test_seed}, len(test_results): {len(test_results)}, len(train_results): {len(train_results)}"
            )
            if STRICT:
                assert False
            continue

        for strategy in results[args.strategy_key].unique():
            strategy_train_results = train_results[
                train_results[args.strategy_key] == strategy
            ]
            strategy_test_results = test_results[
                test_results[args.strategy_key] == strategy
            ]

            best_identifier, best_hyperparams, best_median_auauc = get_best_hyperparams(
                strategy_train_results,
                strategy,
            )

            # Get auauc for test results using best hyperparams
            best_hyperparams_test_results = strategy_test_results[
                strategy_test_results["hyperparam_identifier"] == best_identifier
            ]
            test_auauc = best_hyperparams_test_results[args.auc_metric].values
            # breakpoint()
            assert len(test_auauc) == 1, f"len(test_auauc): {len(test_auauc)}"
            test_auauc = test_auauc[0]

            # Add to results
            all_test_results.append(
                {
                    "strategy": strategy,
                    "best_hyperparams": best_hyperparams,
                    "best_hyperparams_identifier": best_identifier,
                    "best_median_auauc": best_median_auauc,  # median auauc on train seeds
                    "test_auauc": test_auauc,
                    "test_seed": test_seed,
                    "train_seeds": list(train_results["config/run"].unique()),
                    "num_train_seeds": len(train_results["config/run"].unique()),
                }
            )

            test_auaucs_by_strat[strategy].append(test_auauc)

    # Get median auauc across folds
    median_test_auaucs_by_strat = {
        strat: np.median(test_auaucs_by_strat[strat])
        for strat in results[args.strategy_key].unique()
    }

    print("Done.")
    return all_test_results, median_test_auaucs_by_strat


def get_hyperparam_identifier(row):
    """
    Get identifier for hyperparams in row.
    """
    return "-".join([f"{key}_{row[f'config/{key}']}" for key in args.hyperparams])


def get_best_hyperparams(
    results,
    strategy,
):
    """
    Get best hyperparams for strategy according to auc_metric.
    """
    strategy_results = results[results[args.strategy_key] == strategy]

    assert len(strategy_results) > 0, f"len(strategy_results): {len(strategy_results)}"

    # Get the best set of hyperparams according to auc_metric for the strategy, taking median across runs
    # Iterate through all combinations of hyperparams that exist in the data and get the median auauc

    median_auaucs_by_identifier = {}

    # Get median auauc across runs for each hyperparam identifier
    for identifier in results["hyperparam_identifier"].unique():
        # Get results for this identifier
        identifier_results = strategy_results[
            strategy_results["hyperparam_identifier"] == identifier
        ]

        # assert (
        #     len(identifier_results) == 9
        # ), f"identifier: {identifier}, strategy: {strategy}, runs: {identifier_results['config/run'].unique()} != 9, len: {len(identifier_results)}"
        if len(identifier_results) != (NUM_SEEDS - 1):
            print(
                f"identifier: {identifier}, strategy: {strategy}, runs: {identifier_results['config/run'].unique()} != 9, len: {len(identifier_results)}"
            )
            if STRICT:
                assert False
            # continue

        # Get median auauc
        median_auauc = np.median(identifier_results[args.auc_metric].values)
        # Add median auauc to results
        median_auaucs_by_identifier[identifier] = median_auauc

    # Get best hyperparams
    best_identifier = max(
        median_auaucs_by_identifier, key=median_auaucs_by_identifier.get
    )
    best_hyperparams = {
        key: value for key, value in zip(args.hyperparams, best_identifier.split("-"))
    }
    median_auauc = median_auaucs_by_identifier[best_identifier]
    return best_identifier, best_hyperparams, median_auauc


# main
if __name__ == "__main__":
    args = parse_args()
    print("args:", args.data_path)
    results = load_data(args)
    all_test_results, median_test_auaucs_by_strat = get_cross_validation_results(
        results
    )

    results_df = pd.DataFrame(all_test_results)
    results_df.to_csv(args.results_path)
    print("Saved results to:", args.results_path)
    print(results_df.head())

    print()
    print("num_train_seeds:", results_df["num_train_seeds"].value_counts())
    print()

    # Print sorted median auaucs
    print("Median auaucs:")
    for strat, median_auauc in sorted(
        median_test_auaucs_by_strat.items(), key=lambda item: item[1]
    ):
        print(f"{strat}: {median_auauc}")
