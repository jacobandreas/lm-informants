
import wandb
wandb.login()

import pandas as pd 
import wandb
import numpy as np
import json

api = wandb.Api(timeout=300)
entity, project = "lm-informants", "0901_atr"  # set to your entity and project 
runs = api.runs(entity + "/" + project)

from tqdm import tqdm

summary_list, config_list, name_list = [], [], []
mega_df = pd.DataFrame()

all_data = []

try:
    for run_idx, run in tqdm(enumerate(runs), total=len(runs)):
        
        # add all the keys that are logged that you want to download
        keys = [
            "step",
            "auc",
            'strategy_used_is_train',
            ]
        
        end_keys = ['end_stats/mean_auauc']
                        
        if 'Model Eval Logs' in run.summary:
            model_eval_logs_exists = True
        else:
            model_eval_logs_exists = False
        
        # TODO: comment out, hackily setting to not download
#        model_eval_logs_exists=False
        
        # download Model Eval Logs table
        if model_eval_logs_exists:
            
            path = run.summary['Model Eval Logs']['path']
            run.file(path).download(replace=True)

            table = json.load(open(path))
            
            model_eval_logs = pd.DataFrame(data=table['data'], columns=table['columns'])
    #         display(df)
    #         table = run.use_artifact(f"run-{run.id}-Model Eval Logs").get("Model Eval Logs")

    #         model_eval_logs = run.summary['Model Eval Logs']
    #         model_eval_logs = (run.history(keys=['Model Eval Logs'])['Model Eval Logs'][0])
    #         print(model_eval_logs)
        
        history = run.scan_history()
        
        history_df = pd.DataFrame(history)
                    
        # filter ones that were killed
        if run.state != "finished":
            print("filtering run (not finished): ", run.path)
            continue
            
        # get the experiment config
        config = {k: v for k,v in run.config.items()
             if not k.startswith('_')}
        
        exp = {}
        
        # tells us whether strategy_used_is_train_exists was logged as a metric
        strategy_used_is_train_exists = ('strategy_used_is_train' in history_df.columns)
        
        # if key doesn't exist, set to nan (though this shouldn't happen after filtering empty runs)
        exp.update({f"{col}": history_df[~history_df[col].isnull()][col].values if col in history_df.columns else np.nan for col in keys})
        
        # end stats should only be logged once
        end_key_results = {}
        for k in end_keys:
            try:
                val = history_df[~history_df[k].isnull()][k].values
                is_error = False
            except Exception as e:
                print("ERROR FOR:")
                print(run)
                print(e)
                print('continuing...')
                is_error = True
                continue 
            assert len(val) == 1
            end_key_results.update({f"{k}": val[0]})

        if is_error:
            continue
        
        num_steps = len(exp[keys[0]])
        for k in keys:
            if isinstance(exp[k], float) and np.isnan(exp[k]):
                print(f'Warning: key {k} is nan for run: ', run.path)
                # set to list of nans
                exp[k] = [np.nan] * num_steps
                continue
            # only make sure that the non-nan keys have num_steps many entries
            assert len(exp[k]) == num_steps
            
        # convert dict of lists to list of dicts
        results = [dict(zip(exp,t)) for t in zip(*exp.values())]
        
    #     print(results)
        
        for r in results:
            r.update({f"config/{key}": val for key, val in config.items()})
            r.update({f"end_results/{key}": val for key, val in end_key_results.items()})
            r.update({f"strategy": run.name})
            r.update({f"wandb_id": run.path})
            r.update({'tags': run.tags})
            
            if model_eval_logs_exists:
                table_row = model_eval_logs.iloc[int(r['step'])]
                strategy_used = table_row['strategy_for_this_candidate']
                proposed_form = table_row['proposed_form']
                judgment = table_row['judgment']
                features = table_row['features']
                r.update({
                    'strategy_used': strategy_used,
                    'proposed_form': proposed_form,
                    'judgment': judgment,
                    'features': features,
                    })
                
                if strategy_used_is_train_exists:
                    assert r['strategy_used_is_train'] == (strategy_used == 'train'), (f"logged strategy_used_is_train is {r['strategy_used_is_train']}"
                                                                                  f" but strategy_used from table is {strategy_used}"
                                                                                  f" for step {r['step']}")
                
                # if strategy_used_is_train wasn't logged, set it with the table logged at the end
                else:
                    r['strategy_used_is_train'] = (strategy_used == 'train')
            
            # strategy_used_is_train will be nan if BOTH strategy_used_is_train wasn't directly logged AND Model Eval Logs wasn't logged (it defaults to nan if not in the columns when looping through keys)
            # strategy_used will be nan if model_eval_logs_exists = False (i.e. the Model Eval Logs table wasn't logged)
            else:
                r['strategy_used'] = np.nan
            
            
        if run_idx == 0:
            print(results)
                    
        all_data.extend(results)
        
except Exception as e:
    print("Error, writing out temp data")
    pass
#     print(all_data)
        
        
# get status
    
mega_df = pd.DataFrame(all_data)

out_path = './0901_atr_full_1010.csv'
mega_df.to_csv(out_path) 
