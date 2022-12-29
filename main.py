#!/usr/bin/env python3

from analysis.AnalyzeSyntheticData import is_ti
import datasets
from evaluations import evaluate_discrim
import informants
import learners
from logger import Logger
import scorers

import json
import numpy as np
from tqdm import tqdm

np.random.seed(0)

def main():
    dataset = datasets.load_atr_harmony()
    eval_scorer = scorers.HWScorer(dataset, "atr")
    random = np.random.RandomState(0)
    logger = Logger()
    logger.begin()

    train_examples = dataset.data
    random.shuffle(train_examples)
    eval_informant = informants.HWInformant(dataset, eval_scorer)
    informant = eval_informant

    for n_init in [0,16,32,64]:
        for run in range(1):
            for strategy in ["unif"]:
                learner = learners.VBLearner(
                    dataset,
                    strategy=strategy,
                    phoneme_features=eval_informant.scorer.phoneme_features,
                    feature_vocab=eval_informant.scorer.feature_vocab,
                )
                learner.initialize()
                for _ in range(n_init-1):
                    learner.observe(train_examples.pop(0), True, update=True)
                learner.observe(train_examples.pop(0), True, update=True)

                for i in range(100-n_init):
                    train_example = train_examples.pop(0) if strategy == "train" else None
                    candidate = learner.propose(n_candidates=10, train_candidate=train_example)
                    judgment = informant.judge(candidate)
                    learner.observe(candidate, judgment)

                    step_data = {
                        "ent": learner.entropy(candidate),
                        "Step": i+n_init,
                        "Run": run,
                        "Strategy": strategy,
                        "N_Init": n_init,
                        "IsTI": is_ti(str(dataset.vocab.decode(candidate)).replace(",","")),
                        "judgment": judgment,
                        "proposed": candidate,
                    }
                    eval_data = evaluate_discrim(dataset, eval_informant, learner)
                    logger.log_eval(**eval_data, **step_data)
                    logger.log_feats(learner.all_features(), **step_data)


    logger.end()

if __name__ == "__main__":
    main()
