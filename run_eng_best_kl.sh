EXP_DIR=results/0804_english
WANDB_PROJECT=0804_english
TAG=0809_selected,return_true_if_lexicon
NUM_STEPS=500
NUM_CANDS=50

# prior p_all_off of 0.5
PRIOR=0.004595
for LLA in 1.939392
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 10 --feature_type "english" --eval_humans --strategies kl_train_model --feature_type english --num_candidates $NUM_CANDS --tags $TAG --max_updates_observe 1 --max_updates_propose 1 
	#--verbose 
done
