NUM_STEPS=10
STRATEGIES='eig_train_model'

PRIOR=0.004595
LLA=0.970219 

kernprof -l main.py --feature_type english --num_steps ${NUM_STEPS} --num_runs 1 --strategies ${STRATEGIES} --profile_name ${STUB}_${NUM_STEPS} --prior_prob $PRIOR --wandb_project TEST --log_log_alpha_ratio $LLA --max_updates_propose 1 --max_updates_observe 1 --num_candidates 50 
python -m line_profiler main.py.lprof > out.txt
