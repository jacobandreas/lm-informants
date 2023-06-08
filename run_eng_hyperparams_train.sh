EXP_DIR=results/0522_eng
WANDB_PROJECT=0607_english
#TAG=0606_hyperparams_selected
TAG=0606_hyperparams_full
NUM_STEPS=200
NUM_CANDS=50

# prior p_all_off of 0.5
PRIOR=0.000385435812071119
for LLA in 0.818508339363156 1.32655384099565 1.65014245712181 2.13625332091839 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG 
	#--verbose 
done

# prior p_all_off of 0.3
PRIOR=0.000669393655992854
for LLA in 1.72623647509002 2.09445314096582 2.35332338102535 2.76969036320956 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG 
	#--verbose 
done

# prior p_all_off of 0.1
PRIOR=0.00127981720494661
for LLA in 3.15040457368557 3.42914027054695 3.63886266492049 3.99541135890715
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG 
	#--verbose 
done
