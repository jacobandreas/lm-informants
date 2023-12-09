WANDB_PROJECT=linear_alpha_beta_search
WANDB_TEAM=lm-informants
N_SEEDS=2
N_STEPS=100
N_CANDIDATES=100

for ALPHA in 2.5 5 10 20
do
  for BETA in -5 -10 -20 -40
  do
    python eval.py \
      --name ${WANDB_PROJECT} \
      --output_dir ${WANDB_PROJECT} \
      --wandb_team ${WANDB_TEAM} \
      --n_seeds ${N_SEEDS} \
      --n_steps ${N_STEPS} \
      --n_candidates ${N_CANDIDATES} \
      --feature_type atr_harmony \
      --alpha_prior_mu ${ALPHA} \
      --beta_prior_mu ${BETA} \
      --strategies train unif entropy_pred eig_train_model \
      --group_by alpha_prior_mu beta_prior_mu \
      --shuffle_train
  done
done