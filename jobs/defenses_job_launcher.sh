#! /bin/bash

for malicious in {0..60..5}; do
    for defense in "fedavg" "krum" "mkrum" "norm" "cbaa" "distance" "distribution"; do
        
        echo '#!/usr/bin/env bash' > job.sh;
        echo '#SBATCH --account="r260042"' >> job.sh;
        echo '#SBATCH --time=6:00:00' >> job.sh;
        echo '#SBATCH --mem=16G' >> job.sh;
        echo '#SBATCH --constraint=armgpu' >> job.sh;
        echo '#SBATCH --nodes=2' >> job.sh;
        echo '#SBATCH --cpus-per-task=1' >> job.sh;
        echo '#SBATCH --gpus-per-node=1' >> job.sh;
        echo "#SBATCH --job-name \"$malicious $defense\"" >> job.sh;
        echo '#SBATCH --error=output/job.%J.err' >> job.sh;
        echo '#SBATCH --output=output/job.%J.out' >> job.sh;

        echo '' >> job.sh;

        echo 'romeo_load_armgpu_env' >> job.sh;
        echo 'spack load py-pip ^python@3.11.9' >> job.sh;
        echo 'source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate' >> job.sh;

        echo '' >> job.sh;

        echo "python /gpfs/home/griesmax/Federated-Learning/run.py run-defenses --defense $defense --malicious $malicious" >> job.sh;

        sbatch job.sh
    done
done