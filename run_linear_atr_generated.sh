WANDB_PROJECT=linear_atr_generated
WANDB_TEAM=lm-informants
N_STEPS=150
N_CANDIDATES=100

ALPHA_SIGMA=0.01
ALPHA=5
BETA=-5

for SEED in 0 1 2 
do
    python generate_langs.py --seed ${SEED}

    FEATURE_TYPE="atr_lg_"${SEED}

    python eval.py \
      --name ${WANDB_PROJECT} \
      --output_dir ${WANDB_PROJECT} \
      --wandb_team ${WANDB_TEAM} \
      --n_seeds 1 \
      --start_seed ${SEED} \
      --n_steps ${N_STEPS} \
      --n_candidates ${N_CANDIDATES} \
      --feature_type ${FEATURE_TYPE} \
      --alpha_prior_mu ${ALPHA} \
      --beta_prior_mu ${BETA} \
      --alpha_prior_sigma ${ALPHA_SIGMA} \
      --group_by alpha_prior_mu beta_prior_mu \
      --shuffle_train
#      --strategies train unif entropy_pred eig_train_model \
done
