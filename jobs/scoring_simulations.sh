#!/usr/bin/env bash
#SBATCH --account="r260042"
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --constraint=armgpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name "Scoring"
#SBATCH --error=output/job.%J.err
#SBATCH --output=output/job.%J.out

romeo_load_armgpu_env
spack load py-pip ^python@3.11.9

mkdir -p output
source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate

# Scorings
# python run.py run-scoring --run-count 2 \
#     --save-filename distance_scoring \
#     --rounds 5 \
#     --server-scoring true \
#     --metric distance \
#     --threshold 0.4 \
#     --sigma 1,2,3,4,4.5,5,5.5,6,8,10 \
#     --client-count 3 \
#     --epochs 5 \
#     --batch 128 \
#     --fraction 1

# python run.py run-scoring --run-count 2 \
#     --save-filename dataset_scoring \
#     --rounds 5 \
#     --server-scoring true \
#     --metric dataset \
#     --threshold 0.4 \
#     --sigma 1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1 \
#     --client-count 3 \
#     --epochs 5 \
#     --batch 128 \
#     --fraction 1

python run.py run-scoring --run-count 2 \
    --save-filename distribution_scoring \
    --rounds 5 \
    --server-scoring true \
    --metric distribution \
    --threshold 0.4 \
    --bins 20,40,50,70,90,100,110,130,150,160 \
    --client-count 3 \
    --epochs 5 \
    --batch 128 \
    --fraction 1