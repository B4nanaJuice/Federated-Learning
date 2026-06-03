#! /bin/bash

mkdir -p jobs/defense

for malicious in {0..100..5}; do
    for defense in "fedavg" "krum" "mkrum" "norm" "cbaa" "tmean" "rfa" "fltrust" "clra"; do
        
        echo '#!/usr/bin/env bash' > jobs/defense/$defense.$malicious.sh;
        echo '#SBATCH --account="r260042"' >> jobs/defense/$defense.$malicious.sh;
        echo '#SBATCH --time=6:00:00' >> jobs/defense/$defense.$malicious.sh;
        echo '#SBATCH --mem=16G' >> jobs/defense/$defense.$malicious.sh;
        echo '#SBATCH --constraint=armgpu' >> jobs/defense/$defense.$malicious.sh;
        echo '#SBATCH --nodes=2' >> jobs/defense/$defense.$malicious.sh;
        echo '#SBATCH --cpus-per-task=1' >> jobs/defense/$defense.$malicious.sh;
        echo '#SBATCH --gpus-per-node=1' >> jobs/defense/$defense.$malicious.sh;
        echo "#SBATCH --job-name \"$malicious $defense\"" >> jobs/defense/$defense.$malicious.sh;
        echo '#SBATCH --error=output/job.%J.err' >> jobs/defense/$defense.$malicious.sh;
        echo '#SBATCH --output=output/job.%J.out' >> jobs/defense/$defense.$malicious.sh;

        echo '' >> jobs/defense/$defense.$malicious.sh;

        echo 'romeo_load_armgpu_env' >> jobs/defense/$defense.$malicious.sh;
        echo 'spack load py-pip ^python@3.11.9' >> jobs/defense/$defense.$malicious.sh;
        echo 'source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate' >> jobs/defense/$defense.$malicious.sh;

        echo '' >> jobs/defense/$defense.$malicious.sh;

        echo "python /gpfs/home/griesmax/Federated-Learning/run.py server-defense --defense $defense --malicious $malicious" >> jobs/defense/$defense.$malicious.sh;

        sbatch jobs/defense/$defense.$malicious.sh;
    done
done