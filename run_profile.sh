NUM_STEPS=50
STRATEGIES='eig train kl eig_train_model kl_train_model'
# STUB='test'
# python main.py --feature_type atr_harmony --num_steps ${NUM_STEPS} --num_runs 1 --strategies ${STRATEGIES} --profile_name ${STUB}_${STRATEGIES}_${NUM_STEPS} && python view_profiler.py --name ${STUB}_${STRATEGIES}_${NUM_STEPS}

STUB='english_0522'
kernprof -l main.py --feature_type english --num_steps ${NUM_STEPS} --num_runs 1 --strategies ${STRATEGIES} --profile_name ${STUB}_${NUM_STEPS} --prior_prob 0.1 --wandb_project 0522_english --log_log_alpha_ratio 2 
#python view_profiler.py --name ${STUB}_${STRATEGIES}_${NUM_STEPS} 

