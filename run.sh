for LOG in 1
do
	for PRIOR in 0.1
	do
		python main.py --exp_dir results/0508/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project lm-informants_english --num_steps 150 --num_runs 3 --tags "english" --feature_type "english" --eval_humans --warm_start
	done
done
