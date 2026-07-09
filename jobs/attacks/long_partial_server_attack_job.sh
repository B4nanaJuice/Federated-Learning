#!/usr/bin/env bash
#SBATCH --account="r260042"
#SBATCH --time=10:00:00
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
python run.py long-server --partial true