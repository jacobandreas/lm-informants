from main import read_in_blicks, BOUNDARY, eval_auc
import scorers
import datasets
import informants
import learners
from util import entropy, kl_bern

from tqdm import tqdm
import pandas as pd
import itertools

def load_informant_scorer(feature_type):
    lexicon_path = f'data/hw/{feature_type}_lexicon.txt'
    phoneme_feature_path = f'data/hw/{feature_type}_features.txt'
    ngram_feature_path = f'data/hw/{feature_type}_feature_weights.txt'

    print(f'Loading lexicon from:\t{lexicon_path}')
    dataset = datasets.load_lexicon(lexicon_path, min_length=2, max_length=5)

#     You'll probably want to change this line to create your own scorer
    mf_scorer = scorers.MeanFieldScorer(dataset, 
                                        feature_type=feature_type, 
                                        phoneme_feature_file=phoneme_feature_path,
                                       )
    hw_scorer = scorers.HWScorer(dataset, 
                                        feature_type=feature_type, 
                                        phoneme_feature_file=phoneme_feature_path,
                                )

    # Load oracle
    informant = informants.HWInformant(dataset, hw_scorer)
    
    return informant, mf_scorer

# Read in eval dataset

def load_train_dataset(file_name, informant, mf_scorer):
    print("Loading train dataset...")

    # Change this path if you want to specify a different eval dataset
    # eval_dataset_path = f'{feature_type}_test_set.txt'

    # Hacky, but the atr_harmony test set is stored at test_set.csv; the eval dataset names need to be standardized
    df = pd.read_csv(file_name)
    
    items, scores, phonemes, encoded_items, featurized_items, labels, new_costs = [], [], [], [], [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        item = (row['word'].strip().split(' '))
        score = float(row['score'])
        phon = [BOUNDARY] + item + [BOUNDARY]
        encoded = informant.dataset.vocab.encode(phon)
        features = mf_scorer._featurize(encoded).nonzero()[0]
        label = informant.judge(encoded)
        new_cost = informant.cost(encoded)

        items.append(item)
        scores.append(score)
        phonemes.append(phon)
        encoded_items.append(encoded)
        featurized_items.append(features)
        labels.append(label)
        new_costs.append(new_cost)

    # Get dataframe of results
    train_dataset = pd.DataFrame({
        'item': items,
        'label': labels,
        'cost': new_costs,
        'old_cost': scores,
        'encoded': encoded_items,
        'featurized': featurized_items,
    })
    
    test_label_agreement(train_dataset, informant)
    print("Done.")

    return train_dataset

def test_label_agreement(train_dataset, informant):
    """ Sanity check that the labels agree with labels from current oracle """

    print("Testing label agreement...")
    num_judgments_disagree = 0
    num_costs_disagree = 0
    for _, row in tqdm(train_dataset.iterrows(), total=len(train_dataset)):
        encod = row['encoded']
        judgment = row['label']
        informant.set_dataset = {}
        cost = informant.scorer.cost(encod)
        new_judgment = (cost < 2.53)
        
        judgments_agree = (judgment == new_judgment)
        costs_agree = (row['old_cost'] == row['cost'])
        
        if not judgments_agree or not costs_agree:
            print("word:", row['item'])
            print("cost:",cost)
            print("encoded:",encod)
        
        if not judgments_agree:
            print("old judgment:", judgment)
            print("curr judgment:", new_judgment)
            num_judgments_disagree += 1
        if not costs_agree:
            num_costs_disagree += 1
            print("old cost:", row['old_cost'])
            print("new cost:", row['cost'])
    #         assert False

    print("NUM JUDGMENTS DISAGREE:", num_judgments_disagree)
    print("NUM COSTS DISAGREE:", num_costs_disagree)
    print("Done.")
   
import random
import numpy as np


def set_seeds(seed, dataset):
    random.seed(seed)
    np.random.seed(seed)
    dataset.random.seed(seed)

# Read in eval dataset

def load_eval_dataset(informant, mf_scorer):

    # Change this path if you want to specify a different eval dataset
    # eval_dataset_path = f'{feature_type}_test_set.txt'

    # Hacky, but the atr_harmony test set is stored at test_set.csv; the eval dataset names need to be standardized
    df = pd.read_csv('WordsAndScoresFixed_newest.csv')
    
    items = [i.strip().split(' ') for i in df['Word'].values]
    sources = [int(s) for s in df['Source'].values]
    labels = df['Score'].values
   
    phonemes, encoded_items, featurized_items, labels, new_costs = [], [], [], [], []

    for item in tqdm(items):
        phon = ([BOUNDARY]+item+[BOUNDARY])
        phonemes.append(phon)
        encod = (informant.dataset.vocab.encode(phon))
        encoded_items.append(encod)
        featurized_items.append(mf_scorer._featurize(encod).nonzero()[0])
        labels.append(informant.judge(encod))
        new_costs.append(informant.cost(encod))

        # TODO: check label agreemnt?

    """
    # Get phonemes
    phonemes = [[BOUNDARY] + item + [BOUNDARY] for item in tqdm(items)]
    # Encode items
    encoded_items = [informant.dataset.vocab.encode(phon) for phon in tqdm(phonemes)]
    # Featurize items
    featurized_items = [mf_scorer._featurize(encod).nonzero()[0] for encod in tqdm(encoded_items)]
    # Get labels with HW oracle
    # assert same
    labels = [informant.judge(encod) for encod in tqdm(encoded_items)]

    new_costs = [informant.cost(encod) for encod in tqdm(encoded_items)]
    """
    
    # Get dataframe of results
    eval_dataset = pd.DataFrame({
        'item': items,
        'label': labels,
        'encoded': encoded_items,
        'featurized': featurized_items,
        'source': sources,
        'cost': new_costs,
    })

    # only 5 is auc, others are human eval sets
    return eval_dataset[eval_dataset['source']==5] 

def get_auc(scorer, eval_dataset):
    # Learner.cost() is used to get predictions for the test set
    costs = [scorer.cost(encod) for encod in eval_dataset['encoded'].values]
    auc = eval_auc(costs, eval_dataset['label'].values)
    return auc

import os
import wandb

def initialize_hyp(lla, prior, tol, max_updates, dataset, phoneme_feature_path):
    print("Initializing learner...")
    # You may also have to create a slightly modified learner class to wrap around your linear model scorer
    scorer = scorers.MeanFieldScorer(dataset, 
                                     log_log_alpha_ratio=lla,
                                     prior_prob=prior,
                                     feature_type=feature_type,
                                     tolerance=tol,
                                     phoneme_feature_file=phoneme_feature_path,
                                   )
    
    
    
    return scorer

def run(lla, prior, max_updates, tol, train_file, eval_dataset, phoneme_feature_path,
        out_dir='big_batch', 
        wandb_project='1128_big_batch', 
        num_samples=None,
       ):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    sub_dir = os.path.join(out_dir, f'lla={lla}_prior={prior}_max-updates={max_updates}_tol={tol}_num-samples={num_samples}')
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    probs_file = os.path.join(sub_dir, f'probs.npy')
        
    config = {
        'log_log_alpha_ratio': lla,
        'prior_prob': prior,
        'max_updates': max_updates,
        'tolerance': tol,
        'train_file': train_file,
        'num_samples': num_samples,
        'probs_file': probs_file,
    }
    
    wandb_run = wandb.init(config=config, project=wandb_project, reinit=True, entity='lm-informants')

    print(config)
    
    global TRAIN_CACHES

    if train_file in TRAIN_CACHES:
        train_dataset = TRAIN_CACHES[train_file]
    else:
        if train_file == 'WordsAndScoresFixed_newest.csv':
            # TODO: hacky, but load eval dataset if eval file
            train_dataset = load_eval_dataset(informant, mf_scorer)
        else:
            train_dataset = load_train_dataset(train_file, informant, mf_scorer)
        TRAIN_CACHES[train_file] = train_dataset
        
        

    scorer = initialize_hyp(lla, prior, tol, max_updates, informant.dataset, phoneme_feature_path)
    
    print('avg prior:', scorer.probs.mean())
    
    ordered_feats = train_dataset['featurized'].values
    ordered_judgments = train_dataset['label'].values
    ordered_judgments = [1 if j else -1 for j in ordered_judgments]
    
    if num_samples is not None:
        # TODO: setting seed of informant.dataset; is that what we want? 
        # (I think it would only matter for learner.dataset, for getting a train candidate, which we're not using here)
        set_seeds(SEED, informant.dataset)
        ordered_feats, ordered_judgments = zip(*random.sample(list(zip(ordered_feats, ordered_judgments)), num_samples))

    print('# data:', len(ordered_feats))
        
    # TODO: setting seed of informant.dataset; is that what we want? 
    # (I think it would only matter for learner.dataset, for getting a train candidate, which we're not using here)
    set_seeds(SEED, informant.dataset)
    
    # Log distribution over train scores
    table_data = [[s] for s in train_dataset['cost'].values]
    table = wandb.Table(data=table_data, columns=["oracle_costs"])
    wandb.log({'train_oracle_costs': wandb.plot.histogram(table, "oracle_costs",
          title="Train: Distribution of oracle costs", )})
    
    # Log distribution over eval scores
    table_data = [[s] for s in eval_dataset['cost'].values]
#    print(table_data)
    table = wandb.Table(data=table_data, columns=["oracle_costs"])
    wandb.log({'eval_oracle_costs': wandb.plot.histogram(table, "oracle_costs",
          title="Eval: Distribution of oracle costs", )})

    scorer.update(
        ordered_feats, ordered_judgments, 
        max_updates=max_updates,
        verbose=False)
    
    print("Getting auc...")
    auc = get_auc(scorer, eval_dataset)
    print("Done.")

    print("")
    print(f"auc: {auc}")
    
    print('avg posterior:', scorer.probs.mean())
    
    # Log distribution over learned thetas
    table = wandb.Table(data=[[s] for s in scorer.probs], columns=["prob"])
    wandb.log({'learned_probs': wandb.plot.histogram(table, "prob",
          title="Distribution of learned thetas", )})
    
    np.save(probs_file, scorer.probs)
    wandb.save(probs_file)
    # save to probs.py so that it shows up on wandb that way
#    np.save('probs.npy', scorer.probs,)
#    wandb.save('probs.npy')
    print(f"Writing probs to: {probs_file}")
#    # move probs.py file to probs_file
#    os.rename('probs.npy', probs_file)
    
    auc_file_name = os.path.join(sub_dir, 'auc.txt')
    print(f'Writing auc to {auc_file_name}')
    print(f'{auc}', file=open(auc_file_name, 'w'))
    
    wandb.log({'auc': auc})

    p = np.load(probs_file)
    print("mean of loaded probs:", p.mean())

    wandb.log({'mean_learned_probs': p.mean()})

    print("================================")
    
    wandb_run.finish()

if __name__ == "__main__":

    TRAIN_CACHES = {}
    feature_type = 'english'
    SEED=1
    informant, mf_scorer = load_informant_scorer(feature_type)


    # # lla = 0.5
    # lla = 0.522731931474557
    # prior = 0.00138533389897108
    # tol = 0.001/512
    # max_updates=None

#    llas = [0.522731931474557]
#    priors = [0.00138533389897108]
    priors = [0.00240504883318384]
    llas = [5.41687946870128]
    tols = [0.001/512]
    max_updates_lst = [1, None]
    # num_samples_lst = [None, 1000]
#    num_samples_lst = [10, None, 5000, 1000]
    num_samples_lst = [10, None]

    eval_dataset = load_eval_dataset(informant, mf_scorer)

    data_dir = 'data/BabbleRandomStringsEnglish'
    random_strings_file = f'{data_dir}/RandomStringsSubsampledBalanced.csv'
    random_wellformed_file = (f'{data_dir}/RandomWellFormedSyllablesSubsampledBalanced.csv')
    babbled_file = (f'data/MakingOverTrainSet/EnglishOverTrainingData.csv')
    eval_file = 'WordsAndScoresFixed_newest.csv'

#    train_files = [random_strings_file, random_wellformed_file, babbled_file]
#    train_files = [('WordsAndScoresFixed_newest.csv')]
    train_files = [babbled_file, eval_file]
    phoneme_feature_path = f'data/hw/{feature_type}_features.txt'

    for lla, prior, tol, max_updates, num_samples, train_file in itertools.product(llas, priors, tols, max_updates_lst, num_samples_lst, train_files):

        run(lla, prior, max_updates, tol, train_file, eval_dataset, phoneme_feature_path, num_samples=num_samples)
