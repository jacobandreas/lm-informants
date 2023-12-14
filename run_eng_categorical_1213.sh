EXP_DIR=results/1213_english_zipfian
WANDB_PROJECT=1213_english_zipfian
TAG=1214_50_cands
NUM_STEPS=250
NUM_CANDS=50
SEED=$1
STRATEGIES=$2
echo "seed: $SEED";
echo "strategies: $STRATEGIES";

# prior p_all_off of 0.5
PRIOR=0.00227035682449808
LLA=5.25792364149749
python main.py --exp_dir ${EXP_DIR}/prior=${PRIOR}-log_ratio=${LLA}-seed=${SEED} \
	--log_log_alpha_ratio ${LLA} \
	--prior_prob ${PRIOR} \
	--wandb_project $WANDB_PROJECT \
	--num_steps $NUM_STEPS \
	--num_runs 1 \
	--start_run ${SEED} \
	--feature_type "english" \
	--eval_humans \
	--strategies $STRATEGIES \
	--feature_type english \
	--num_candidates $NUM_CANDS \
	--tags $TAG \
	--max_updates_observe 1 \
	--max_updates_propose 1 \
	--use_zipfian \
	--train_expect_type poisson_samples
