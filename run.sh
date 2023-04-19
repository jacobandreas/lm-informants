for LOG in 1 5 20 
do
	for PRIOR in 0.2 0.5
	do
		for TOL in 0.05
		do	
		python main.py --exp_dir results/0419/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project lm-informants_0419 --num_steps 75 --num_runs 1 --tolerance ${TOL}
		done
	done
done
