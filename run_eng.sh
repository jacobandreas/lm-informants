EXP_DIR=results/0522_eng

for LOG in 1 
do
	for PRIOR in 0.0035
	do
#		kernprof -l main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project 0522_english --num_steps 2 --num_runs 1 --feature_type "english" --eval_humans --strategies train eig_train_model train unif --feature_type english 
		kernprof -l main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --num_steps 50 --num_runs 1 --feature_type "english" --eval_humans --strategies eig_train_model train unif --wandb_project 0522_english 
		#--verbose 
	done
done
