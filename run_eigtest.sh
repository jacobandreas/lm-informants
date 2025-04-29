#!/bin/bash
#SBATCH --partition=cpl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-user=amanirmk@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --output=eigtest_categorical1
set -e
cd /om/user/amanirmk/projects/new/lm-informants/
source /om/user/amanirmk/miniconda/bin/activate
conda activate informants

python main.py \
  --num_runs=3 \
  --num_steps=100 \
  --num_candidates=1 \
  --feature_type="atr_lg_eigtest" \
  --strategies unif eig \
  --wandb_project "eigtest_categorical"