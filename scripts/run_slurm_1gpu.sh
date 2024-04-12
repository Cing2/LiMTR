#!/bin/bash
#SBATCH --job-name=limtr_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00


# load modules

PY_ARGS=${@:1}
echo $PY_ARGS

srun python src/trainLight.py local=server ${PY_ARGS}
