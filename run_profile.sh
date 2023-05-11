NUM_STEPS=5
STRATEGIES='eig'

#python main.py --feature_type atr_harmony --num_steps ${NUM_STEPS} --num_runs 1 --strategies ${STRATEGIES} --profile_name atr_${STRATEGIES}_${NUM_STEPS} && python view_profiler.py --name atr_${STRATEGIES}_${NUM_STEPS}

python main.py --feature_type english --num_steps ${NUM_STEPS} --num_runs 1 --strategies ${STRATEGIES} --profile_name english_${STRATEGIES}_${NUM_STEPS} && python view_profiler.py --name english_${STRATEGIES}_${NUM_STEPS}

