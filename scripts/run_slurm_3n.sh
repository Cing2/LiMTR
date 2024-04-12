#!/bin/bash
#SBATCH --job-name=limtr_train
#SBATCH --nodes=3             # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --gpus=12
#SBATCH --partition=gpu
#SBATCH --time=1-0

# load modules


export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

PY_ARGS=${@:1}
echo $PY_ARGS

srun python src/trainLight.py local=server trainer=ddp-3 ${PY_ARGS}
