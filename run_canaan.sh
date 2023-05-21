EXP_DIR=results/0510

for LOG in .5 1 2 4 8
do
	for PRIOR in .05 
	do
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project maybefinal_four --num_steps 150 --num_runs 10 --tags "maybefinal_four" --feature_type "atr_harmony" --eval_humans --strategies entropy entropy_pred train unif
	
	done
done
