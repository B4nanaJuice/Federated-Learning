#! /bin/bash

mkdir -p jobs/defense

for mal in {0..100..5}; do
    for defense in "tmean" "rfa" "fltrust"; do
        
        echo '#!/usr/bin/env bash' > jobs/defense/$defense.$mal.sh;
        echo '#SBATCH --account="r260042"' >> jobs/defense/$defense.$mal.sh;
        echo '#SBATCH --time=6:00:00' >> jobs/defense/$defense.$mal.sh;
        echo '#SBATCH --mem=16G' >> jobs/defense/$defense.$mal.sh;
        echo '#SBATCH --constraint=armgpu' >> jobs/defense/$defense.$mal.sh;
        echo '#SBATCH --nodes=2' >> jobs/defense/$defense.$mal.sh;
        echo '#SBATCH --cpus-per-task=1' >> jobs/defense/$defense.$mal.sh;
        echo '#SBATCH --gpus-per-node=1' >> jobs/defense/$defense.$mal.sh;
        echo "#SBATCH --job-name \"$mal $defense\"" >> jobs/defense/$defense.$mal.sh;
        echo '#SBATCH --error=output/job.%J.err' >> jobs/defense/$defense.$mal.sh;
        echo '#SBATCH --output=output/job.%J.out' >> jobs/defense/$defense.$mal.sh;

        echo '' >> jobs/defense/$defense.$mal.sh;

        echo 'romeo_load_armgpu_env' >> jobs/defense/$defense.$mal.sh;
        echo 'spack load py-pip ^python@3.11.9' >> jobs/defense/$defense.$mal.sh;
        echo 'source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate' >> jobs/defense/$defense.$mal.sh;

        echo '' >> jobs/defense/$defense.$mal.sh;

        echo "python /gpfs/home/griesmax/Federated-Learning/run.py run-defenses --defense $defense --malicious $mal" >> jobs/defense/$defense.$mal.sh;

        sbatch jobs/defense/$defense.$mal.sh;
    done
done