EXP_DIR=results/0522_eng
WANDB_PROJECT=0610_english_new_tolerance
#TAG=0606_hyperparams_selected
TAG=0606_hyperparams_full
NUM_STEPS=200
NUM_CANDS=50
TOLERANCE=0.00000195312
STRATEGIES=unif
MIN_LENGTH=2
MAX_LENGTH=8

# prior p_all_off of 0.5
PRIOT=0.000692907009547478
for LLA in -0.582669213511331 0.0996555117688712 0.522188203482090 0.876593342814206 1.07490225924691 1.33270866303582
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies $STRATEGIES --feature_type english --num_candidates $NUM_CANDS --tags $TAG --tolerance $TOLERANCE --min_length $MIN_LENGTH --max_length $MAX_LENGTH 
	#--verbose 
done

# prior p_all_off of 0.3
PRIOR=0.00120324831985153
for LLA in -0.0323131016905313 0.771581138033756 1.11235973004718 1.49906804786627
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies $STRATEGIES --feature_type english --num_candidates $NUM_CANDS --tags $TAG --tolerance $TOLERANCE --min_length $MIN_LENGTH --max_length $MAX_LENGTH 
	#--verbose 
done

# prior p_all_off of 0.1
PRIOR=0.00229993617744669
for LLA in 0.968894497787593 1.93773271312620 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies $STRATEGIES --feature_type english --num_candidates $NUM_CANDS --tags $TAG --tolerance $TOLERANCE --min_length $MIN_LENGTH --max_length $MAX_LENGTH
	#--verbose 
done
