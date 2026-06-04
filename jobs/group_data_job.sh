#!/usr/bin/env bash
#SBATCH --account="r260042"
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --constraint=armgpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name "Group data"
#SBATCH --comment "Job for grouping data from different simulations"
#SBATCH --error=output/job.%J.err
#SBATCH --output=output/job.%J.out

romeo_load_armgpu_env
spack load py-pip ^python@3.11.9

mkdir -p output
source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate

for scoring in "distance" "dataset"; do
    for decay in "root" "log"; do
        for partial in "partial" "total"; do
            python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "${decay}_$scoring $partial"
        done

        for malicious in {0..100..5}; do
            python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "${decay}_$scoring $malicious"
        done
    done
done