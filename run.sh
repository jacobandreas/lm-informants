for LOG in 5 10 2
do
	for PRIOR in 0.5 0.2
	do
#		python main.py --exp_dir results/0403/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project lm-informants_0403 --num_steps 150 --num_runs 2 --feature_type english
		python main.py --exp_dir results/0405/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project lm-informants_0405 --num_steps 150 --num_runs 2 
	done
done
