for LOG in 2
do
	for PRIOR in 0.5
	do
		for tolerance in 0.001
		do
			python main.py --exp_dir results/0322_kiku/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project lm-informants_0322_toy --num_steps 3 --num_runs 1 --feature_type kiku --no-eval_humans --no-shuffle_train  --converge_type symmetric --verbose --tolerance ${tolerance}
		done
	done
done
