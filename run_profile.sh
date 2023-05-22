NUM_STEPS=2
STRATEGIES='eig'
# STUB='test'
# python main.py --feature_type atr_harmony --num_steps ${NUM_STEPS} --num_runs 1 --strategies ${STRATEGIES} --profile_name ${STUB}_${STRATEGIES}_${NUM_STEPS} && python view_profiler.py --name ${STUB}_${STRATEGIES}_${NUM_STEPS}

STUB='english_no_multiprocessing'
kernprof -l main.py --feature_type english --num_steps ${NUM_STEPS} --num_runs 1 --strategies ${STRATEGIES} --profile_name ${STUB}_${STRATEGIES}_${NUM_STEPS} && python view_profiler.py --name ${STUB}_${STRATEGIES}_${NUM_STEPS}

