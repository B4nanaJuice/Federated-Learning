#! /bin/bash

mkdir -p jobs/scoring

for scoring in "distance" "dataset"; do
    for partial in "true" "false"; do
        for decay in "root" "log"; do
        
            echo '#!/usr/bin/env bash' > jobs/scoring/$scoring.$partial.$decay.sh;
            echo '#SBATCH --account="r260042"' >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo '#SBATCH --time=6:00:00' >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo '#SBATCH --mem=16G' >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo '#SBATCH --constraint=armgpu' >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo '#SBATCH --nodes=2' >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo '#SBATCH --cpus-per-task=1' >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo '#SBATCH --gpus-per-node=1' >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo "#SBATCH --job-name \"$scoring $partial\"" >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo '#SBATCH --error=output/job.%J.err' >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo '#SBATCH --output=output/job.%J.out' >> jobs/scoring/$scoring.$partial.$decay.sh;

            echo '' >> jobs/scoring/$scoring.$partial.$decay.sh;

            echo 'romeo_load_armgpu_env' >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo 'spack load py-pip ^python@3.11.9' >> jobs/scoring/$scoring.$partial.$decay.sh;
            echo 'source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate' >> jobs/scoring/$scoring.$partial.$decay.sh;

            echo '' >> jobs/scoring/$scoring.$partial.$decay.sh;

            echo "python /gpfs/home/griesmax/Federated-Learning/run.py client-decay --save-filename $scoring --partial $partial --metric $scoring --decay $decay" >> jobs/scoring/$scoring.$partial.$decay.sh;

            sbatch jobs/scoring/$scoring.$partial.$decay.sh;
        done
    done
done