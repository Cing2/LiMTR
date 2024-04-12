#!/bin/bash
#SBATCH --job-name=limtr_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=18
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=1-0

# load modules

PY_ARGS=${@:1}
echo $PY_ARGS

srun python src/trainLight.py local=server trainer=ddp ${PY_ARGS}
