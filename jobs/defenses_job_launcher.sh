#! /bin/bash

mkdir -p jobs/defense

for partial in "true" "false"; do
    for defense in "tmean" "rfa" "fltrust"; do
        
        echo '#!/usr/bin/env bash' > jobs/defense/$defense.$partial.sh;
        echo '#SBATCH --account="r260042"' >> jobs/defense/$defense.$partial.sh;
        echo '#SBATCH --time=6:00:00' >> jobs/defense/$defense.$partial.sh;
        echo '#SBATCH --mem=16G' >> jobs/defense/$defense.$partial.sh;
        echo '#SBATCH --constraint=armgpu' >> jobs/defense/$defense.$partial.sh;
        echo '#SBATCH --nodes=2' >> jobs/defense/$defense.$partial.sh;
        echo '#SBATCH --cpus-per-task=1' >> jobs/defense/$defense.$partial.sh;
        echo '#SBATCH --gpus-per-node=1' >> jobs/defense/$defense.$partial.sh;
        echo "#SBATCH --job-name \"$partial $defense\"" >> jobs/defense/$defense.$partial.sh;
        echo '#SBATCH --error=output/job.%J.err' >> jobs/defense/$defense.$partial.sh;
        echo '#SBATCH --output=output/job.%J.out' >> jobs/defense/$defense.$partial.sh;

        echo '' >> jobs/defense/$defense.$partial.sh;

        echo 'romeo_load_armgpu_env' >> jobs/defense/$defense.$partial.sh;
        echo 'spack load py-pip ^python@3.11.9' >> jobs/defense/$defense.$partial.sh;
        echo 'source /gpfs/home/griesmax/Federated-Learning/venv/bin/activate' >> jobs/defense/$defense.$partial.sh;

        echo '' >> jobs/defense/$defense.$partial.sh;

        echo "python /gpfs/home/griesmax/Federated-Learning/run.py run-defenses --defense $defense --partial $partial" >> jobs/defense/$defense.$partial.sh;

        sbatch jobs/defense/$defense.$partial.sh;
    done
done