WANDB_PROJECT=1024_atr_generated

for RUN in 0 1 2 3 4 5 6 7 8 9 
do
#	for LOG in 0.1 0.25 0.5 1 2 4 8 
	for LOG in 1 2 4 8
	do
		for PRIOR in 0.001 0.025 0.03125 0.05 0.1 0.2 0.35
		do
			for MAX_UPDATES in None 1
			do
			python main.py --exp_dir results/1010/warm/batch_prior=${PRIOR}-log_ratio=${LOG} \
			--log_log_alpha_ratio ${LOG} \
			--prior_prob ${PRIOR} \
			--wandb_project $WANDB_PROJECT \
			--num_steps 150 --num_runs 1 --tags "1011_optimal_prior" \
			--feature_type "atr_lg_0" --eval_humans \
			--max_updates_observe $MAX_UPDATES \
			--max_updates_propose $MAX_UPDATES \
			--start_run ${RUN} \
	#		--strategies eig 
			done
		done
	done
done
