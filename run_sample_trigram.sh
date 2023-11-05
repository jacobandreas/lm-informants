WANDB_PROJECT=1018_atr_harmony_sample_trigram

for RUN in 0 1 2 3 4 5 6 7 8 9 
do
	for LOG in 0.1 0.25 0.5 1 2 4 8 
	do
#		for PRIOR in 0.025 0.05 0.1 0.2 0.35
		for PRIOR in 0.03125
		do
			for MAX_UPDATES in None 1
			do
			python main.py --exp_dir results/1010/warm/batch_prior=${PRIOR}-log_ratio=${LOG} \
			--log_log_alpha_ratio ${LOG} \
			--prior_prob ${PRIOR} \
			--wandb_project $WANDB_PROJECT \
			--num_steps 150 --num_runs 1 --tags "1018" \
			--feature_type "atr_harmony" --eval_humans \
			--max_updates_observe $MAX_UPDATES \
			--max_updates_propose $MAX_UPDATES \
			--start_run ${RUN} \
			--use_trigram_oracle \
			--trigram_oracle_seed 0
			done
		done
	done
done
