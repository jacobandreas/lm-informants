WANDB_PROJECT=linear_english_001
WANDB_TEAM=lm-informants
N_SEEDS=3
N_STEPS=150
N_CANDIDATES=100

python eval.py \
  --name ${WANDB_PROJECT} \
  --output_dir ${WANDB_PROJECT} \
  --wandb_team ${WANDB_TEAM} \
  --n_seeds ${N_SEEDS} \
  --n_steps ${N_STEPS} \
  --n_candidates ${N_CANDIDATES} \
  --feature_type english \
  --shuffle_train