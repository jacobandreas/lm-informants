EXP_DIR=results/0511

for LOG in 8
do
	for PRIOR in .35
	do
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project maybefinal_four --num_steps 150 --num_runs 10 --tags "maybefinal_four" --feature_type "atr" --eval_humans --strategies eig_train_model 
	done
done
