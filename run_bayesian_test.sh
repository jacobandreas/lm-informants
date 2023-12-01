WANDB_PROJECT=testing2
N_SEEDS=3
N_STEPS=100
N_CANDIDATES=100

for BETA in -10 -5 -1
do
    for USE_MEAN in "--use_mean" ""
    do
    python eval.py \
    --name ${WANDB_PROJECT} \
    --output_dir ${WANDB_PROJECT} \
    --n_seeds ${N_SEEDS} \
    --n_steps ${N_STEPS} \
    --n_candidates ${N_CANDIDATES} \
    ${USE_MEAN} \
    --beta_prior_mu ${BETA} \
    --label_run_by use_mean beta_prior_mu \
    --strategies entropy_pred train \
    --shuffle_train
    done
done