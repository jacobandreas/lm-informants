WANDB_PROJECT=0901_atr

for LOG in 0.25 
do
	for PRIOR in 0.1 0.2 0.35
	do
		for MAX_UPDATES in None 1
		do
		python main.py --exp_dir results/0610_test/warm/batch_prior=${PRIOR}-log_ratio=${LOG} \
		--log_log_alpha_ratio ${LOG} \
		--prior_prob ${PRIOR} \
		--wandb_project $WANDB_PROJECT \
		--num_steps 150 --num_runs 3 --tags "0906_OM" \
		--feature_type "atr_harmony" --eval_humans \
		--max_updates_observe $MAX_UPDATES \
		--max_updates_propose $MAX_UPDATES \
	        --start_run 3	
#		--strategies eig 
		done
	done
done
