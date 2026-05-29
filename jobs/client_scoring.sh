#! /bin/bash

mkdir -p jobs/scoring

for scoring in "distance" "dataset" "distribution" "similarity"; do
    for partial in "true" "false"; do
        
        echo '#!/usr/bin/env bash' > jobs/defense/$scoring.$partial.sh;
        echo '#SBATCH --account="r260042"' >> jobs/defense/$scoring.$partial.sh;
        echo '#SBATCH --time=6:00:00' >> jobs/defense/$scoring.$partial.sh;
        echo '#SBATCH --mem=16G' >> jobs/defense/$scoring.$partial.sh;
        echo '#SBATCH --constraint=armgpu' >> jobs/defense/$scoring.$partial.sh;
        echo '#SBATCH --nodes=2' >> jobs/defense/$scoring.$partial.sh;
        echo '#SBATCH --cpus-per-task=1' >> jobs/defense/$scoring.$partial.sh;
        echo '#SBATCH --gpus-per-node=1' >> jobs/defense/$scoring.$partial.sh;
        echo "#SBATCH --job-name \"$scoring $partial\"" >> jobs/defense/$scoring.$partial.sh;
        echo '#SBATCH --error=output/job.%J.err' >> jobs/defense/$scoring.$partial.sh;
        echo '#SBATCH --output=output/job.%J.out' >> jobs/defense/$scoring.$partial.sh;

        echo '' >> jobs/defense/$scoring.$partial.sh;

        echo 'romeo_load_armgpu_env' >> jobs/defense/$scoring.$partial.sh;
        echo 'spack load py-pip ^python@3.11.9' >> jobs/defense/$scoring.$partial.sh;
        echo 'source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate' >> jobs/defense/$scoring.$partial.sh;

        echo '' >> jobs/defense/$scoring.$partial.sh;

        echo "python /gpfs/home/griesmax/Federated-Learning/run.py client-scoring --save-filename $scoring --partial $partial --metric $scoring" >> jobs/defense/$scoring.$partial.sh;

        sbatch jobs/defense/$scoring.$partial.sh;
    done
done