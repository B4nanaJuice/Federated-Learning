#! /bin/bash

mkdir -p jobs/scoring

for scoring in "distance" "dataset" "distribution" "similarity"; do
    for partial in "true" "false"; do
        
        echo '#!/usr/bin/env bash' > jobs/scoring/$scoring.$partial.sh;
        echo '#SBATCH --account="r260042"' >> jobs/scoring/$scoring.$partial.sh;
        echo '#SBATCH --time=6:00:00' >> jobs/scoring/$scoring.$partial.sh;
        echo '#SBATCH --mem=16G' >> jobs/scoring/$scoring.$partial.sh;
        echo '#SBATCH --constraint=armgpu' >> jobs/scoring/$scoring.$partial.sh;
        echo '#SBATCH --nodes=2' >> jobs/scoring/$scoring.$partial.sh;
        echo '#SBATCH --cpus-per-task=1' >> jobs/scoring/$scoring.$partial.sh;
        echo '#SBATCH --gpus-per-node=1' >> jobs/scoring/$scoring.$partial.sh;
        echo "#SBATCH --job-name \"$scoring $partial\"" >> jobs/scoring/$scoring.$partial.sh;
        echo '#SBATCH --error=output/job.%J.err' >> jobs/scoring/$scoring.$partial.sh;
        echo '#SBATCH --output=output/job.%J.out' >> jobs/scoring/$scoring.$partial.sh;

        echo '' >> jobs/scoring/$scoring.$partial.sh;

        echo 'romeo_load_armgpu_env' >> jobs/scoring/$scoring.$partial.sh;
        echo 'spack load py-pip ^python@3.11.9' >> jobs/scoring/$scoring.$partial.sh;
        echo 'source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate' >> jobs/scoring/$scoring.$partial.sh;

        echo '' >> jobs/scoring/$scoring.$partial.sh;

        echo "python /gpfs/home/griesmax/Federated-Learning/run.py client-scoring --save-filename $scoring --partial $partial --metric $scoring" >> jobs/scoring/$scoring.$partial.sh;

        sbatch jobs/scoring/$scoring.$partial.sh;
    done
done