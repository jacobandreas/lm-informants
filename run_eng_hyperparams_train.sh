EXP_DIR=results/0522_eng
WANDB_PROJECT=0607_english
#TAG=0606_hyperparams_selected
TAG=0606_hyperparams_full
NUM_STEPS=200
NUM_CANDS=50

# prior p_all_off of 0.5
PRIOR=0.00138533389897108
for LLA in 0.818807677680882 1.32677908055553 1.65032042172503 2.13636980699140 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG 
	#--verbose 
done

# prior p_all_off of 0.3
PRIOR=0.00240504883318384
for LLA in 1.72665652700838 2.09477871455906 2.35358707233326 2.76987047901232
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG 
	#--verbose 
done

# prior p_all_off of 0.1
PRIOR=0.00459458264847305
for LLA in 3.15104773856974 3.42965456729417 3.63928993623763 3.99571598252335 
do
	python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LLA} --log_log_alpha_ratio ${LLA} --prior_prob ${PRIOR} --wandb_project $WANDB_PROJECT --num_steps $NUM_STEPS --num_runs 1 --feature_type "english" --eval_humans --strategies train --feature_type english --num_candidates $NUM_CANDS --tags $TAG 
	#--verbose 
done
