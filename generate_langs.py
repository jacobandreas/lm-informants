from main import read_in_blicks, BOUNDARY
import scorers
import datasets
import informants
import random

import pandas as pd
import argparse
import os
import numpy as np


def make_words(syllables,num_words,list_to_exclude=[]):
    """ Returns a list of lists of syllables, ex.:[['ti'], ['qe', 'ka', 'qE'], ['ki', 'qI', 'qi', 'qa', 'qi', 'ke', 'ta'], ...] """
    word_list = []
    while len(word_list) < num_words:
        syllable_list = []
        length = np.random.poisson(2)
        for _ in range(length) if length > 0 else range(1):
            next_syllable = random.choice(syllables)
            syllable_list.append(next_syllable)
        word = " ".join(syllable_list)
        if word not in word_list and word not in list_to_exclude:
            word_list.append(word)
    items = [w.strip().split(' ') for w in word_list]
    return items 

def classify_word(list_of_forbidden_features,word):
    phonemes = [BOUNDARY] + word + [BOUNDARY]
    # Encode items
    encoded_item = dataset.vocab.encode(phonemes)
    features = mf_scorer._featurize(encoded_item).nonzero()[0]
    # some features are in forbidden features
    if len(set.intersection(set(features), set(list_of_forbidden_features))) != 0:
        return False
    else:
        return True

def make_a_train_and_test_set(num_words,list_of_forbidden_features,syllables):
    words = make_words(syllables, num_words)
    goods = []
    bads = []
    for word in words: # word is a list like ['ti'] or  ['qe', 'ka', 'qE'], etc.
        if classify_word(list_of_forbidden_features,word):
            goods.append((" ".join(word),True))
        else:
            bads.append((" ".join(word),False))
    random.shuffle(goods)
    train = goods[:len(goods)//2]
    len_test_set = min(len(train),len(bads))
    test_goods = random.sample(goods[len(goods)//2:],len_test_set)
    test_bads = random.sample(bads,len_test_set)
    return train, test_goods+test_bads


def make_a_language(numerical_features, syllables, num_words, num_bad_features):
    bad_features = random.sample(numerical_features, num_bad_features)
    train, test = make_a_train_and_test_set(num_words, bad_features, syllables)
    return train, test, bad_features

def write_out_a_language(train, test, bad_features, seed):
    out_dir = f'data/generated_langs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Write lexicon
    lexicon_file = f'{out_dir}/atr_lg_{seed}_lexicon.txt'
    print(f'Writing lexicon to: {lexicon_file}')
    with open(lexicon_file, 'w') as f:
        for (item, label) in train:
            f.write(f'{item}\n')
   
    # Write eval set
    test_file = f'{out_dir}/atr_lg_{seed}_test_set.csv'
    print(f'Writing test set to: {test_file}')
    test_items = [i[0] for i in test]
    test_labels = [i[1] for i in test]
    eval_dataset = pd.DataFrame({'item': test_items, 'label': test_labels})
    eval_dataset.to_csv(test_file)

    feature_file = f'{out_dir}/atr_lg_{seed}_trigram_features.txt'
    print(f'Writing features to: {feature_file}')
    test_items = [i[0] for i in test]
    with open(feature_file, 'w') as f:
        for b in bad_features:
            f.write(f'{b}\n')



def main(num_words, num_bad_features,numerical_features,syllables,seed):
    feature_type = 'atr_harmony'

    print("working on language seed", seed)
    random.seed(seed)
    train, test, bad_features = make_a_language(numerical_features,syllables, num_words, num_bad_features)
    #print(train)
    write_out_a_language(train, test, bad_features, seed)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()

    print('args:', args)

    syllables = [
    'pa',
    'ta',
    'ka',
    'pi',
    'ti',
    'ki',
    'pe',
    'te',
    'ke',
    'pE',
    'tE',
    'kE',
    'pI',
    'tI',
    'kI',
    'qI',
    'qi',
    'qe',
    'qE',
    'qa'

    ]

    num_words = 1000
    num_bad_features = 16
    
    feature_type = 'atr_harmony'
    # Change these paths if you want to specify a different set of features
    lexicon_path = f'data/hw/{feature_type}_lexicon.txt'
    phoneme_feature_path = f'data/hw/{feature_type}_features.txt'
    ngram_feature_path = f'data/hw/{feature_type}_feature_weights.txt'

    print(f'Loading lexicon from:\t{lexicon_path}')

    global dataset
    dataset = datasets.load_lexicon(lexicon_path, min_length=2, max_length=5)


    global mf_scorer
    mf_scorer = scorers.MeanFieldScorer(dataset, 
                                        feature_type=feature_type, 
                                        phoneme_feature_file=phoneme_feature_path,
                                       )
    
    numerical_features = [i for i in mf_scorer.ngram_features.values()]
    main(num_words, num_bad_features, numerical_features, syllables, args.seed)
