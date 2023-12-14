EXP_DIR=results/1213_english_zipfian
WANDB_PROJECT=1213_english_zipfian
TAG=test
NUM_STEPS=250
NUM_CANDS=100
SEED=0

# prior p_all_off of 0.5
PRIOR=0.00227035682449808
LLA=5.25792364149749
python main.py --exp_dir ${EXP_DIR}/prior=${PRIOR}-log_ratio=${LLA} \
	--log_log_alpha_ratio ${LLA} \
	--prior_prob ${PRIOR} \
	--wandb_project $WANDB_PROJECT \
	--num_steps $NUM_STEPS \
	--num_runs 1 \
	--start_run ${SEED} \
	--feature_type "english" \
	--eval_humans \
	--strategies eig_train_mixed train unif \
	--feature_type english \
	--num_candidates $NUM_CANDS \
	--tags $TAG \
	--max_updates_observe 1 \
	--max_updates_propose 1 \
	--use_zipfian \
	--train_expect_type poisson_samples
