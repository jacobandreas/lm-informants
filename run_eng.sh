EXP_DIR=results/0522_eng

for LOG in 8 
do
	for PRIOR in .01
	do
		kernprof -l main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project 0522_english --num_steps 150 --num_runs 3 --feature_type "english" --eval_humans --strategies train unif kl kl_train_model eig eig_train_model --feature_type english 
	done
done
