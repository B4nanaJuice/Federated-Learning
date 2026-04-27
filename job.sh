#!/usr/bin/env bash
#SBATCH --account="r260042"
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --constraint=armgpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

romeo_load_armgpu_env
spack load python@3.11.9/esusxhd

mkdir -p output
source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate

python run.py run-simulation --run-count 10 --max-rounds 20 --total-clients 20 --malicious-client-count 0 --client-fraction 0.5 --epochs 15 --save-filename "clean_run"