#!/usr/bin/env bash
#SBATCH --account="r260042"
#SBATCH --time=1:00:00
#SBATCH --mem=5G
#SBATCH --constraint=armgpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

romeo_load_armgpu_env
# spack load python@3.11.9/oxq4fb7
spack load cuda@12.6.2

source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate

mkdir -p logs

python run.py run-simulation \
    --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 0 \
    --client-fraction 0.5 \
    --epochs 15 \
    --min-clients 0