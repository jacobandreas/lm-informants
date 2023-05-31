EXP_DIR=results/0510

#for LOG in 1 
#do
#	for PRIOR in .35
#	do
#		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project maybefinal_four --num_steps 150 --num_runs 10 --tags "maybefinal_four" --feature_type "atr_harmony" --eval_humans --strategies kl kl_train_model kl_train_history kl_train_mixed eig eig_train_model eig_train_history eig_train_mixed
#	done
#done

#for LOG in .5 1 2 4 8
for LOG in 4 8
do
	for PRIOR in .1 .2 .35
	do
		python main.py --exp_dir ${EXP_DIR}/warm/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project maybefinal_four --num_steps 150 --num_runs 10 --tags "maybefinal_four" --feature_type "atr_harmony" --eval_humans --strategies kl kl_train_model kl_train_history kl_train_mixed eig eig_train_model eig_train_history eig_train_mixed
	done
done
