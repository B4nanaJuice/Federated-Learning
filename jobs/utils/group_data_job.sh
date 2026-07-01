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

for attack in "clean" "5_data" "20_data" "5_gaussian_weights" "20_gaussian_weights" "20_sign_flip" "20_gradient_amplification" "partial" "total"; done
    python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "$attack"
done

for defense in "fedavg" "krum" "mkrum" "norm" "cbaa" "tmean" "rfa" "fltrust" "clra"; do 
    for malicious in {0..100..5}; do
        python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "$defense $malicious.0"
    done
done