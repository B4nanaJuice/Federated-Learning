#!/usr/bin/env bash
#SBATCH --account="r260042"
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --constraint=armgpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name "Distribution scoring"
#SBATCH --error=output/job.%J.err
#SBATCH --output=output/job.%J.out

romeo_load_armgpu_env
spack load py-pip ^python@3.11.9

mkdir -p output
source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate

# Attacker that will poison only last layer of the broadcasted model
python run.py run-scoring --run-count 10 \
    --save-filename distribution_scoring \
    --rounds 15 \
    --server-scoring true \
    --metric distribution \
    --threshold 0.4 \
    --bins 20,40,50,70,90,100,110,130,150,160 \
    --client-count 5 \
    --epochs 15 \
    --batch 128 \
    --fraction 1