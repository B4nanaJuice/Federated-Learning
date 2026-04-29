#!/usr/bin/env bash
#SBATCH --account="r260042"
#SBATCH --time=5:30:00
#SBATCH --mem=16G
#SBATCH --constraint=armgpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name "5% model"
#SBATCH --comment "Clean run with 10 simulations, 20 total clients and 4 malicious clients, 10 clients per round for 20 rounds"
#SBATCH --error=output/job.%J.err
#SBATCH --output=output/job.%J.out

romeo_load_armgpu_env
spack load py-pip ^python@3.11.9

mkdir -p output
source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate

# 5% malicious clients attacking their data continuously
python run.py run-simulation --run-count 10 \
    --max-rounds 20 \
    --total-clients 20 \
    --malicious-client-count 1 \
    --client-fraction 0.5 \
    --epochs 15 \
    --save-filename "5%_clients_model" \
    --client-attack-rate "lambda x: True"