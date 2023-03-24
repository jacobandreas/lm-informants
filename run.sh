for LOG in 2 5 10 
do
	for PRIOR in 0.2 0.5
	do
		python main.py --exp_dir results/0324/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project lm-informants_0324 --num_steps 150 --num_runs 2
	done
done
