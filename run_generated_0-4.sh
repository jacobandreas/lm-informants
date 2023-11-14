WANDB_PROJECT=1106_all_atr_generated

for SEED in 0 1 2 3 4 
do
	# Generate languages
	python generate_langs.py --seed ${SEED}

#	for LOG in 0.1 0.25 0.5 1 2 4 8 
#	# ignoring 4/8 for now because those were worse

	FEATURE_TYPE="atr_lg_"${SEED}

	for LOG in 0.1 0.25 0.5 1 2 
	do
		for PRIOR in 0.001 0.025 0.03125 0.05 0.1 0.2 0.35
		do
			for MAX_UPDATES in None 1
			do
			python main.py \
			--exp_dir results/1106_all_atr_generated/prior=${PRIOR}-log_ratio=${LOG}-seed=${SEED}-updates=${MAX_UPDATES} \
			--log_log_alpha_ratio ${LOG} \
			--prior_prob ${PRIOR} \
			--wandb_project $WANDB_PROJECT \
			--num_steps 150 --num_runs 1 \
			--feature_type ${FEATURE_TYPE} --eval_humans \
			--max_updates_observe $MAX_UPDATES \
			--max_updates_propose $MAX_UPDATES \
			--start_run ${SEED} \
	#		--strategies eig 
#			--tags "" \
			done
		done
	done
done
