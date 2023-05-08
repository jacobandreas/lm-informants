for LOG in 2
do
	for PRIOR in 0.1
	do
		python main.py --exp_dir results/0504/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project lm-informants_maybefinal_three --num_steps 150 --num_runs 10 --tags "maybefinal_three" --feature_type "atr_harmony" --eval_humans --warm_start
		python main.py --exp_dir results/0504/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project lm-informants_maybefinal_three --num_steps 150 --num_runs 10 --tags "maybefinal_three" --feature_type "atr_harmony" --eval_humans
	done
done
