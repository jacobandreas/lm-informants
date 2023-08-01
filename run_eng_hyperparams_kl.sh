EXP_DIR=results/0801_english
WANDB_PROJECT=0801_english
TAG=0801_hyperparam_sweep,fixed_eval_sets
NUM_STEPS=200
NUM_CANDS=50

# prior p_all_off of 0.5
PRIOR=0.001385
# commenting out bc already ran, restarted
for LLA in -0.582269524436029 0.100113 0.522736 0.87727 1.075687 1.333741 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies kl_train_model --feature_type english --num_candidates $NUM_CANDS --tags $TAG --max_updates_observe 1 --max_updates_propose 1 
	#--verbose 
done

# prior p_all_off of 0.3
PRIOR=0.002405
for LLA in -0.03161 0.77243 1.113339 1.500334 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies kl_train_model --feature_type english --num_candidates $NUM_CANDS --tags $TAG --max_updates_observe 1 --max_updates_propose 1 
	#--verbose 
done

# prior p_all_off of 0.1
PRIOR=0.004595
for LLA in 0.970219 1.939392 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies kl_train_model --feature_type english --num_candidates $NUM_CANDS --tags $TAG --max_updates_observe 1 --max_updates_propose 1 
done
