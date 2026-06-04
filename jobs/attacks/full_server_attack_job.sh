#!/usr/bin/env bash
#SBATCH --account="r260042"
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --constraint=armgpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name "Chaos"
#SBATCH --comment "One-shot total attack on server's global model"
#SBATCH --error=output/job.%J.err
#SBATCH --output=output/job.%J.out

romeo_load_armgpu_env
spack load py-pip ^python@3.11.9

mkdir -p output
source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate

# Attacker that has total access on server's model
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 0 \
    --client-fraction 0.5 \
    --epochs 15 \
    --attacked-server True \
    --save-filename "total_takeover" \
    --server-attack-rate "lambda x: x == 8"