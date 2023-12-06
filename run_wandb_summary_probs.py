""" 
Download the probs.py files!
"""

from tqdm import tqdm
import numpy as np
import os
import pandas as pd 
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("lm-informants/1106_all_atr_generated")

summary_list, config_list, name_list = [], [], []
tag_list = []
# a dict of lists where keys are steps and values are the associated probs at that step for all runs (in order of runs)
probs_list = {i: [] for i in range(151)}
for run in tqdm(runs[:10]): 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)
    tag_list.append(run.tags)
    
    root = 'temp_files'
    num_files_found = 0

    if run.state != "finished":
        print(f"Skipping run {run.id} because status is {run.state}")
        continue

    for f in run.files():
        if f.name.endswith(f'.npy'):
            f.download(root=root, replace=True)
            f_path = os.path.join(root, f.name)

            stub = f.name.split('/')[-1]
            i = int(stub.split('.')[0])
            probs = np.load(f_path)

            probs_list[i].append(probs)
            num_files_found += 1

    assert num_files_found == 151, f'only found {num_files_found} files'

results = ({
    "summary": summary_list,
    "config": config_list,
    "name": name_list,
    'tags': tag_list,
    })

results.update({f'probs_{k}': v for k,v in probs_list.items()})

for r, v in results.items():
    print(r, len(v))

runs_df = pd.DataFrame(results)

#runs_df.to_csv("1024_atr_generated_summary_1101.csv")
#runs_df.to_csv("1106_all_atr_generated_summary_1108.csv")
runs_df.to_csv("1106_all_atr_generated_summary_1129_with_probs.csv")
