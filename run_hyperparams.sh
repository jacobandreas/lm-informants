EXP_DIR=results/0508

for LOG in 1
do
	for PRIOR in 0.1
	do
		# for now only warm
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project remaining_hyperparams --num_steps 150 --num_runs 10 --tags "maybefinal_three" --feature_type "atr_harmony" --eval_humans --warm_start --strategies train unif kl_train_model entropy entropy_pred eig_train_model eig
	done
done

for LOG in 2 
do
	for PRIOR in 0.2 0.5
	do
		# for now only warm
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project remaining_hyperparams --num_steps 150 --num_runs 10 --tags "maybefinal_three" --feature_type "atr_harmony" --eval_humans --warm_start --strategies kl_train_model eig_train_history eig_train_mixed eig_train_model kl_train_history kl_train_mixed 
	done
done

for LOG in 1 
do
	for PRIOR in 0.2 
	do
		# for now only warm
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project remaining_hyperparams --num_steps 150 --num_runs 10 --tags "maybefinal_three" --feature_type "atr_harmony" --eval_humans --warm_start --strategies eig_train_model kl_train_model 
	done
done

for LOG in 4 
do
	for PRIOR in 0.1 0.2 0.5
	do
		# for now only warm
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project remaining_hyperparams --num_steps 150 --num_runs 10 --tags "maybefinal_three" --feature_type "atr_harmony" --eval_humans --warm_start --strategies eig_train_model
	done
done
