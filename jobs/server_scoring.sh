#! /bin/bash

mkdir -p jobs/scoring

for scoring in "distance" "dataset" "distribution" "similarity"; do
    for malicious in {0..100..5}; do
        
        echo '#!/usr/bin/env bash' > jobs/scoring/$scoring.$malicious.sh;
        echo '#SBATCH --account="r260042"' >> jobs/scoring/$scoring.$malicious.sh;
        echo '#SBATCH --time=6:00:00' >> jobs/scoring/$scoring.$malicious.sh;
        echo '#SBATCH --mem=16G' >> jobs/scoring/$scoring.$malicious.sh;
        echo '#SBATCH --constraint=armgpu' >> jobs/scoring/$scoring.$malicious.sh;
        echo '#SBATCH --nodes=2' >> jobs/scoring/$scoring.$malicious.sh;
        echo '#SBATCH --cpus-per-task=1' >> jobs/scoring/$scoring.$malicious.sh;
        echo '#SBATCH --gpus-per-node=1' >> jobs/scoring/$scoring.$malicious.sh;
        echo "#SBATCH --job-name \"$malicious $scoring\"" >> jobs/scoring/$scoring.$malicious.sh;
        echo '#SBATCH --error=output/job.%J.err' >> jobs/scoring/$scoring.$malicious.sh;
        echo '#SBATCH --output=output/job.%J.out' >> jobs/scoring/$scoring.$malicious.sh;

        echo '' >> jobs/scoring/$scoring.$malicious.sh;

        echo 'romeo_load_armgpu_env' >> jobs/scoring/$scoring.$malicious.sh;
        echo 'spack load py-pip ^python@3.11.9' >> jobs/scoring/$scoring.$malicious.sh;
        echo 'source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate' >> jobs/scoring/$scoring.$malicious.sh;

        echo '' >> jobs/scoring/$scoring.$malicious.sh;

        echo "python /gpfs/home/griesmax/Federated-Learning/run.py server-scoring --save-filename $scoring --malicious $malicious --metric $scoring" >> jobs/scoring/$scoring.$malicious.sh;

        sbatch jobs/scoring/$scoring.$malicious.sh;
    done
done