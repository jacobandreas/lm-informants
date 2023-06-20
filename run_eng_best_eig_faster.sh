EXP_DIR=results/0612_eng
WANDB_PROJECT=0610_english_new_tolerance
#TAG=0612_hyperparams_selected
TAG=0612_hyperparams_selected_faster
#TAG=0606_hyperparams_full
NUM_STEPS=500
NUM_CANDS=50
TOLERANCE=0.00000195312
STRATEGIES=eig_train_model

# with only 1 update in propose

# TOP 2 BEST HYPERPARAMS for unif
# prior p_all_off of 0.1
PRIOR=0.004595
for LLA in 0.970219 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies $STRATEGIES --feature_type english --num_candidates $NUM_CANDS --tags $TAG,no_recompute_train,set_feats --tolerance $TOLERANCE --max_updates_propose 1 --max_updates_observe 1
	#--verbose 
done
