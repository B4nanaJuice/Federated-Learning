#!/usr/bin/env bash
#SBATCH --account="r260042"
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --constraint=armgpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name "Partial server"
#SBATCH --comment "Partial attack on server's global model"
#SBATCH --error=output/job.%J.err
#SBATCH --output=output/job.%J.out

romeo_load_armgpu_env
spack load py-pip ^python@3.11.9

mkdir -p output
source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate

# Attacker that will poison only last layer of the broadcasted model
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 0 \
    --client-fraction 0.5 \
    --epochs 15 \
    --attacked-server True \
    --save-filename "partial_corruption" \
    --server-attack-rate "lambda x: x in [7, 8, 12, 14]" \
    --partial_attack True