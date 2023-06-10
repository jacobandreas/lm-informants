for LOG in 0.5 
do
	for PRIOR in 0.1
	do
		python main.py --exp_dir results/0610_test/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project 0610_test --num_steps 50 --num_runs 1 --tags "new" --feature_type "atr_harmony" --eval_humans --strategies eig_train_model --verbose 
	done
done
