EXP_DIR=results/0804_english
WANDB_PROJECT=0804_english
TAG=0804_hyperparam_sweep,return_true_if_lexicon
NUM_STEPS=200
NUM_CANDS=50

for MAX_UPDATES in 1 None
do
	### CALIBRATED: i.e. use same # of features as kl_train_model to compute prior_prob

	# prior p_all_off of 0.5
	PRIOR=0.001385
	for LLA in 3.37379282024428 3.64450190689220 3.84947003210775 4.19987456939477
	do
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG --max_updates_observe ${MAX_UPDATES} 
	done

	# prior p_all_off of 0.3
	PRIOR=0.002405
	for LLA in 5.41687946870128 5.64240070601870 5.81942728026585 6.13231740872503
	do
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG --max_updates_observe ${MAX_UPDATES} 
	done

	# TODO: hyperparams for prior p_all_off won't calculate for some reason
	
	#########################################################
		
	### NOT CALIBRATED: i.e. use same # of features as in computing target posterior to compute prior_prob

	# prior p_all_off of 0.5
	PRIOR=0.000385435812071119
	for LLA in 0.818508339363158 1.32655384099565 1.65014245712180 2.13625332091839
	do
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG --max_updates_observe ${MAX_UPDATES} 
	done

	# prior p_all_off of 0.3
	PRIOR=0.000669393655992854
	for LLA in 1.72623647509002 2.09445314096582 2.35332338102534 2.76969036320956 
	do
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG --max_updates_observe ${MAX_UPDATES} 
	done
	
	# prior p_all_off of 0.1
	PRIOR=0.00127981720494661
	for LLA in 3.15040457368557 3.42914027054695 3.63886266492048 3.99541135890715 
	do
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG --max_updates_observe ${MAX_UPDATES} 
	done
done
