#! /bin/bash

mkdir -p jobs/offline

for scoring in "distance" "dataset" "distribution" "similarity"; do
        
    echo '#!/usr/bin/env bash' > jobs/offline/offline_$scoring.sh;
    echo '#SBATCH --account="r260042"' >> jobs/offline/offline_$scoring.sh;
    echo '#SBATCH --time=6:00:00' >> jobs/offline/offline_$scoring.sh;
    echo '#SBATCH --mem=16G' >> jobs/offline/offline_$scoring.sh;
    echo '#SBATCH --constraint=armgpu' >> jobs/offline/offline_$scoring.sh;
    echo '#SBATCH --nodes=2' >> jobs/offline/offline_$scoring.sh;
    echo '#SBATCH --cpus-per-task=1' >> jobs/offline/offline_$scoring.sh;
    echo '#SBATCH --gpus-per-node=1' >> jobs/offline/offline_$scoring.sh;
    echo "#SBATCH --job-name \"off $scoring\"" >> jobs/offline/offline_$scoring.sh;
    echo '#SBATCH --error=output/job.%J.err' >> jobs/offline/offline_$scoring.sh;
    echo '#SBATCH --output=output/job.%J.out' >> jobs/offline/offline_$scoring.sh;

    echo '' >> jobs/offline/offline_$scoring.sh;

    echo 'romeo_load_armgpu_env' >> jobs/offline/offline_$scoring.sh;
    echo 'spack load py-pip ^python@3.11.9' >> jobs/offline/offline_$scoring.sh;
    echo 'source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate' >> jobs/offline/offline_$scoring.sh;

    echo '' >> jobs/offline/offline_$scoring.sh;

    echo "python /gpfs/home/griesmax/Federated-Learning/run.py offline --metric $scoring" >> jobs/offline/offline_$scoring.sh;

    sbatch jobs/offline/offline_$scoring.sh;
done