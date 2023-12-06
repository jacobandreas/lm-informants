import wandb
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from main import cost_to_prob

file_map = {
    'WordsAndScoresFixed_newest.csv': 'eval',
    'data/BabbleRandomStringsEnglish/RandomStringsSubsampledBalanced.csv': 'random_strings',
    'data/MakingOverTrainSet/EnglishOverTrainingData.csv': 'hw_babble',
    'data/BabbleRandomStringsEnglish/RandomWellFormedSyllablesSubsampledBalanced.csv': 'random_wellformed',
}

inverse_file_map = {v:k for k,v in file_map.items()}

def get_exp(df, lla, prior, max_updates, train_file_short, num_samples=None):
    temp = df[
        (df['log_log_alpha_ratio']==lla) & 
        (df['prior_prob']==prior) & 
        (df['max_updates']==max_updates) & 
        (df['train_file_short']==train_file_short) & 
        (df['num_samples']==num_samples)
    ]

    # display(temp)

    if len(temp) != 1:
        display(temp)
    assert len(temp) == 1, f"Found {len(temp)} rows matching the query"
    # print("AUC:", temp['auc'])

    return temp.iloc[0]

def print_dict(d):
    """ Pretty print a dictionary """
    for k, v in d.items():
        print(f"{k}: {v}")

def get_wandb_runs(project='lm-informants/1128_big_batch'):

    # Set your W&B API key (you can find it in your W&B account settings)
    wandb.login()


    # Get all runs from the project
    api = wandb.Api()
    runs = api.runs(project)

    # Initialize lists to store data
    data = []

    # Iterate over runs
    for run in tqdm(runs):
        run_id = run.id

        # Download run artifacts
    #     run.download()

        # Get run metrics
        metrics = run.history()

        tags = run.tags
        assert len(tags) <= 1

        # Get run config
        config = run.config

        # Append data to the list
        d = ({
            "run_id": run_id,
            "metrics": metrics,
        })
        d.update(config)
        d['config'] = config
        d.update(run.summary._json_dict)

        if tags != []:
            d.update({'tag': tags[0]})

        else:
            d.update({'tag': None})

        # probs = load_probs(run)

        # Check if status succeeded
        if run.state != "finished":
            print(f"Skipping run {run_id} because status is {run.state}")
            continue

        # An alternative way to download npy files
        num_files_found = 0
        root = 'temp_files'
        for f in run.files():
            if f.name.endswith('.npy'):
                # print(f)
                f.download(root=root, replace=True)
                f_path = os.path.join(root, f.name)
                # print(f_path)
                num_files_found += 1

                probs = np.load(f_path)
                
                d.update({'probs': probs, 'probs_mean': probs.mean()})
                data.append(d)

        assert num_files_found == 1, f'Found {num_files_found} npy files for run {run_id}'

        

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    df = df.dropna(subset=['auc'], axis=0)
    df = df.fillna("None")


    # Save DataFrame to a CSV file
#     df.to_csv("wandb_runs.csv", index=False)

    df['train_file_short'] = df.apply(lambda row: file_map[row['train_file']], axis=1)

    return df

def get_probs_file(config, base_dir='big_batch'):
    sub_dir = f'lla={config["log_log_alpha_ratio"]}_prior={config["prior_prob"]}_max-updates={config["max_updates"]}_tol={config["tolerance"]}_num-samples={config["num_samples"]}'
    f = os.path.join(base_dir, sub_dir, 'probs.npy')
    return f

def load_probs(run, project_dir='lm-informants/1114_big_batch'):
    f = get_probs_file(run.config)
    print("Loading probs from file at: ", f)
    f_obj = wandb.restore(f, run_path=f'{project_dir}/{run.id}')
    # f_obj is a Text.io object
    # Write the contents of f_obj to temp 'probs.npy' file so that can call np.load() with it
    return np.fromstring(f_obj.read())

def plot_learned_weights_compare(probs, title=None):
    # TODO: For the trigram features that have *multiple* phonemes in the first slot, should we break that into two separate trigrams?
    """ Plots the values of the learned feature weights probs against the true oracle weights 
    """

    learned_weights = []
    oracle_weights = []

    informant, mf_scorer = load_informant_scorer("english")
    
    indices_to_feats = {feat_idx: mf_scorer.feature_vocab.decode(feat) for feat, feat_idx in mf_scorer.ngram_features.items()}
    indices_to_feats_encoded = {feat_idx: feat for feat, feat_idx in mf_scorer.ngram_features.items()}

    # Define dictionary mapping from feature to index for learned features
    learned_features_to_weights = {}
    for feat_idx, weight in enumerate(probs):
        decoded_feat = indices_to_feats[feat_idx]
        learned_features_to_weights[decoded_feat] = weight

    # Iterate througph probs, and if the feature is in the oracle, append to learned_weights
    for (feat_template, weight) in informant.scorer.ngram_features[3]:
        # print(feat_template)
        # print(weight)
        decoded = (informant.scorer.pp_feature(feat_template))
        # print()

        decoded_tuple = tuple([d[1:-1] for d in decoded.split(' ')])
        if len(decoded_tuple) != 3:
            # print("continuing because len != 3")
            # print(decoded)
            # print(decoded_tuple)
            continue
        else:
            # print(decoded_tuple)
            # print(learned_features_to_weights.keys())
            if decoded_tuple in learned_features_to_weights:
                learned_weights.append(learned_features_to_weights[decoded_tuple])
                oracle_weights.append(weight)
    
    # Plot
    print(f"# overlapping weights: {len(learned_weights)}/{len(informant.scorer.ngram_features[3])+len(informant.scorer.ngram_features[2])}")
    plt.scatter(oracle_weights, learned_weights, alpha=0.5)
    plt.xlabel("Oracle weights")
    plt.ylabel("Learned weights")
    
    if title is None:
        title = ""
    title += ("Oracle weights vs. learned weights")

    plt.title(title)
    plt.show()
    
def plot_auc(df, max_updates=1):
    
    m = file_map

    temp = df[df['max_updates']!=max_updates]


    title = f'max_updates=None'

    temp['train_name'] = temp.apply(lambda row: m[row['train_file']], axis=1)

    # Choose the grouping key (either "num_samples" or "train_file")
    grouping_key = "num_samples"

    # Automatically create a color palette based on unique values in the other key
    palette = sns.color_palette("husl", n_colors=len(temp[grouping_key].unique()))

    # Create a bar plot
    ax = sns.barplot(
        x=grouping_key,
        y="auc",
        hue="train_name" if grouping_key == "num_samples" else "num_samples",
        data=temp,
        palette=palette,
        order=[10.0, 1000.0, 5000.0, 7000.0, "None"],  # Specify the order of x-axis values

    )
    ax.legend(bbox_to_anchor=(1.05, 0), loc='lower center', borderaxespad=0.)

    plt.title(title)
    # Show the plot
    plt.show()


def plot_learned_weights(probs, title=None):
    """ Plot the distribution of learned feature weights in probs, but also show which features are in the oracle.
    Plot two histograms: one for the learned features that are in the oracle features, and one for the learned features that are not in the oracle features.
    """

    feats_in_oracle = []
    feats_not_in_oracle = []
    
    informant, mf_scorer = load_informant_scorer("english")
    
    indices_to_feats = {feat_idx: mf_scorer.feature_vocab.decode(feat) for feat, feat_idx in mf_scorer.ngram_features.items()}
    indices_to_feats_encoded = {feat_idx: feat for feat, feat_idx in mf_scorer.ngram_features.items()}

    # Define dictionary mapping from feature to weight for oracle features
    oracle_features_to_weights = {}
    for (feat_template, weight) in informant.scorer.ngram_features[3]:
        decoded = (informant.scorer.pp_feature(feat_template))
        decoded_tuple = tuple([d[1:-1] for d in decoded.split(' ')])
        if len(decoded_tuple) != 3:
            # print("continuing because len != 3")
            # print(decoded)
            # print(decoded_tuple)
            continue
        else:
            oracle_features_to_weights[decoded_tuple] = weight

    for feat_idx, weight in enumerate(probs):
        decoded_feat = indices_to_feats[feat_idx]

        # Figure out if in oracle
        if decoded_feat in oracle_features_to_weights:
            feats_in_oracle.append(weight)
        else:
            feats_not_in_oracle.append(weight)


    # Plot
    # Create subplots, one for features in oracle, one for features not in oracle
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(feats_in_oracle, bins=40)
    axs[0].set_title("Learned weights for features in oracle")
    axs[0].set_xlabel("Learned weights")
    axs[0].set_ylabel("Frequency")
    # annotate the bars of the histogram with values
    for rect in axs[0].patches:
        height = rect.get_height()
        if height > 0:
            axs[0].annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), 
            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
            rotation=90)

    axs[0].set_ylim([0, 15])

    axs[1].hist(feats_not_in_oracle, bins=20)
    axs[1].set_title("Learned weights for features not in oracle")
    axs[1].set_xlabel("Learned weights")
    axs[1].set_ylabel("Frequency")
    # annotate the bars of the histogram with values
    for rect in axs[1].patches:
        height = rect.get_height()
        if height > 0:
            axs[1].annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), 
            textcoords="offset points", ha='center', va='bottom', rotation=90)
    axs[1].set_ylim([0, 55000])

    if title is None:
        title = ""
    title += ("Distribution of learned weights")
    fig.suptitle(title, fontsize=16, y=1.03)

    plt.show()

def plot_eval_costs_by_label(probs, config, eval_dataset, title=None, costs=None):

    if costs is None:
        # create a dummy scorer
        scorer = initialize_hyp(lla, prior_prob, tol, max_updates, informant.dataset, 'data/hw/english_features.txt')
        scorer.probs = probs
        # get costs for eval items
        costs = [scorer.cost(encod) for encod in eval_dataset['encoded'].values]

    # plot two histograms in two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # plot histogram of costs for items labeled 1
    costs_1 = [costs[i] for i, label in enumerate(eval_dataset['label'].values) if label == 1]
    axs[0].hist(costs_1, bins=20)
    axs[0].set_title("Eval costs for items labeled 1")
    axs[0].set_xlabel("Eval costs")
    axs[0].set_ylabel("Frequency")
    axs[0].set_ylim([0, 1700])
    # annotate the bars of the histogram with values
    for rect in axs[0].patches:
        height = rect.get_height()
        if height > 0:
            axs[0].annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), 
            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
            rotation=90)

    # plot histogram of costs for items labeled 0
    costs_0 = [costs[i] for i, label in enumerate(eval_dataset['label'].values) if label == 0]
    axs[1].hist(costs_0, bins=20)
    axs[1].set_title("Eval costs for items labeled 0")
    axs[1].set_xlabel("Eval costs")
    axs[1].set_ylabel("Frequency")
    axs[1].set_ylim([0, 700])
    # annotate the bars of the histogram with values
    for rect in axs[1].patches:
        height = rect.get_height()
        if height > 0:
            axs[1].annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), 
            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
            rotation=90)

    if title is None:
        title = ""
    title += (f"Eval costs for items labeled 1 vs. 0")
    fig.suptitle(title, fontsize=16)
    plt.show()

# import roc_curve
from sklearn.metrics import auc, roc_curve

# Create function to get roc curve from probs
def get_roc_curve(probs, eval_dataset, costs=None, label="", do_show=True):
    # Create scorer
    if costs is None:
        scorer = initialize_hyp(lla, prior_prob, tol, max_updates, informant.dataset, 'data/hw/english_features.txt')
        scorer.probs = probs
        # Get costs for eval items
        costs = [scorer.cost(encod) for encod in eval_dataset['encoded'].values]
    label_probs = [cost_to_prob(c) for c in costs]

    # Get roc curve
    fpr, tpr, thresholds = roc_curve(eval_dataset['label'].values, label_probs)
    plt.plot(fpr, tpr, label=label)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    if do_show:
        plt.show()

def get_classification_accuracy(probs, eval_dataset, threshold=0.5, costs=None):
    # Plot classification accuracy as a function of threshold
    if costs is None:
        scorer = initialize_hyp(lla, prior_prob, tol, max_updates, informant.dataset, 'data/hw/english_features.txt')
        scorer.probs = probs
        # Get costs for eval items
        costs = [scorer.cost(encod) for encod in eval_dataset['encoded'].values]
    label_probs = [cost_to_prob(c) for c in costs]
    label_preds = [1 if p >= threshold else 0 for p in label_probs]
    gold_labels = eval_dataset['label'].values
    assert all([l in [0, 1] for l in gold_labels])
    acc = sum([1 if p == l else 0 for p, l in zip(label_preds, gold_labels)]) / len(gold_labels)
    return acc


def compute_alpha(lla):
    return np.exp(lla) / (1 + np.exp(lla))

def get_log_likelihood(lla, eval_dataset, costs):
    # alpha is prob that the label is noisy
    alpha = compute_alpha(lla)
    print("alpha:", alpha)
    label_probs = [cost_to_prob(c) for c in costs]
    gold_labels = eval_dataset['label'].values
    assert all([l in [0, 1] for l in gold_labels])
    log_likelihood = 0
    for p_true, l in zip(label_probs, gold_labels):
        if l == 1:
            log_likelihood += np.log(p_true*alpha + (1-p_true)*(1-alpha))
        else:
            log_likelihood += np.log(p_true*(1-alpha) + (1-p_true)*alpha)
    return log_likelihood

def get_average_prob_true(costs, alpha):
    # returns the average probability of the true label across dataset 
    
    probs_true = [cost_to_prob(c) for c in costs]

    probs_true_with_noise = [p_true*alpha + (1-p_true)*(1-alpha) for p_true in probs_true]

    mean = np.mean(probs_true_with_noise)
    std = np.std(probs_true_with_noise)

    return mean, std

def plot_histogram_scores(df, title=None):

    # Assuming your DataFrame is named df and has a column named 'score'
    # For example, df = pd.DataFrame({'score': [85, 90, 88, 92, 78, 95, 87, 88, 90]})

    bin_size = 0.5
    # Plotting a histogram
    
    min_cost = math.floor(min(df['cost']))
    max_cost = math.ceil(max(df['cost']))
    
    bins=np.arange(min_cost, max_cost + bin_size, bin_size)
    fig = plt.figure(figsize=(len(bins)/2, 2))
    
    plt.hist(df['cost'], bins=bins, color='blue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Score')
    plt.xticks(bins)
    plt.ylabel('Frequency')
    if title is not None:
        plt.title(title)

    # Display the plot
    plt.show()
    
def plot_histogram_labels(df, title=None):

    # Assuming your DataFrame is named df and has a column named 'score'
    # For example, df = pd.DataFrame({'score': [85, 90, 88, 92, 78, 95, 87, 88, 90]})

    # Count the occurrences of each label
    label_counts = df['label'].value_counts()

    # Plotting a bar plot
    label_counts.plot(kind='bar', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Label')
    plt.ylabel('Count')
    if title is not None:
        plt.title(title)
    # Display the plot
    plt.show()