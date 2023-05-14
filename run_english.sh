EXP_DIR=results/0512

for LOG in .5 
do
	for PRIOR in .2
	do
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project english --num_steps 150 --num_runs 3 --tags "english" --feature_type "english" --eval_humans --strategies eig_train_model train unif
	done
done
