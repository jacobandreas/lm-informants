WANDB_PROJECT=linear_alpha_sigma_search
WANDB_TEAM=lm-informants
N_SEEDS=3
N_STEPS=100
N_CANDIDATES=100

for SIGMA in 0.01 0.1 1 5 10 20
do
  python eval.py \
    --name ${WANDB_PROJECT} \
    --output_dir ${WANDB_PROJECT} \
    --wandb_team ${WANDB_TEAM} \
    --n_seeds ${N_SEEDS} \
    --n_steps ${N_STEPS} \
    --n_candidates ${N_CANDIDATES} \
    --feature_type atr_harmony \
    --alpha_prior_sigma ${SIGMA} \
    --strategies train unif entropy_pred eig_train_model \
    --group_by alpha_prior_sigma seed \
    --shuffle_train
done