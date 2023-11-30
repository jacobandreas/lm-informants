from tqdm import tqdm
import pandas as pd 
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("lm-informants/1106_all_atr_generated")
#runs = api.runs("lm-informants/1024_atr_generated")

summary_list, config_list, name_list = [], [], []
tag_list = []
for run in tqdm(runs): 
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

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list,
    'tags': tag_list,
    })

#runs_df.to_csv("1024_atr_generated_summary_1101.csv")
runs_df.to_csv("1106_all_atr_generated_summary_1108.csv")
