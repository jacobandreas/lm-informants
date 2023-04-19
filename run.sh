for LOG in 1 5 20 
do
	for PRIOR in 0.2 0.5
	do
		python main.py --exp_dir results/0419/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project lm-informants_0419 --num_steps 150 --num_runs 1
	done
done
