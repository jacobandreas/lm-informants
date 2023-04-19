for LOG in 20 
do
	for PRIOR in 0.5 0.2
	do
		for SYLLABLES in 3 4 10
		do
			python main.py --exp_dir results/0404_kiku/batch_prior=${PRIOR}-log_ratio=${LOG} --log_log_alpha_ratio ${LOG} --prior_prob ${PRIOR} --wandb_project kiku --num_steps 3 --num_runs 1 --feature_type kiku --no-eval_humans --no-shuffle_train  --converge_type symmetric --verbose --lexicon_file data/hw/kiku_lexicon_${SYLLABLES}.txt
		done
	done
done
