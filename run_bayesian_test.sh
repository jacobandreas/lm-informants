WANDB_PROJECT=beta_alpha_search
WANDB_TEAM=lm-informants
N_SEEDS=3
N_STEPS=100
N_CANDIDATES=100

for BETA in 2 4 8 16
do
    for ALPHA in 1 2 4 8 16
    do
    python eval.py \
    --name ${WANDB_PROJECT} \
    --wandb_team ${WANDB_TEAM} \
    --output_dir ${WANDB_PROJECT} \
    --n_seeds ${N_SEEDS} \
    --n_steps ${N_STEPS} \
    --n_candidates ${N_CANDIDATES} \
    --beta_prior_mu ${BETA} \
    --alpha_prior_mu ${ALPHA} \
    --group_by alpha_prior_mu beta_prior_mu \
    --shuffle_train
    done
done