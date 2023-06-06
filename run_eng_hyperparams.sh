EXP_DIR=results/0522_eng
WANDB_PROJECT=0522_english
TAG=0606_hyperparams
NUM_STEPS=20
NUM_CANDS=50

# prior p_all_off of 0.5
PRIOR=0.001385
for LLA in -2.12616 -0.58226 0.100113 0.522736 0.87727 1.075687 1.333741 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies eig_train_model unif --feature_type english --num_candidates $NUM_CANDS --tags $TAG 
	#--verbose 
done

# prior p_all_off of 0.3
PRIOR=0.002405
for LLA in -0.03161 0.77243 1.113339 1.500334 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies eig_train_model unif --feature_type english --num_candidates $NUM_CANDS --tags $TAG 
	#--verbose 
done

# prior p_all_off of 0.1
PRIOR=0.004595
for LLA in 0.970219 1.939392 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies eig_train_model unif --feature_type english --num_candidates $NUM_CANDS --tags $TAG 
	#--verbose 
done
