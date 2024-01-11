EXP_DIR=results/1214_english_zipfian
WANDB_PROJECT=1214_english_zipfian
TAG=canaan
NUM_STEPS=200
NUM_CANDS=100

# prior p_all_off of 0.5
PRIOR=0.00227035682449808
LLA=1.47488617047509
#5.25792364149749
python main.py --exp_dir ${EXP_DIR}/prior=${PRIOR}-log_ratio=${LLA}-seed=${SEED} \
	--log_log_alpha_ratio ${LLA} \
	--prior_prob ${PRIOR} \
	--wandb_project $WANDB_PROJECT \
	--num_steps $NUM_STEPS \
	--num_runs 1 \
	--start_run 3 \
	--feature_type "english" \
	--eval_humans \
	--strategies eig_train_mixed \
	--feature_type english \
	--num_candidates $NUM_CANDS \
	--tags $TAG \
	--max_updates_observe 1 \
	--max_updates_propose 1 \
	--use_zipfian \
	--train_expect_type poisson_samples
