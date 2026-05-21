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

for mal in {0..100..5}; do
    for defense in "tmean" "rfa" "fltrust"; do
        echo "Grouping data for defense $defense and $mal percent attackers"
        python /gpfs/home/griesmax/Federated-Learning/run.py group-data --run-count 10 --save-filename "$defense $mal"
    done
done