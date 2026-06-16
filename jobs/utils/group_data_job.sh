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

for defense in "fedavg" "krum" "mkrum" "norm" "cbaa" "rfa" "tmean" "fltrust" "clra"; do
    python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "$defense partial"
    python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "$defense total"
done

python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "clean"
python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "5.0_data"
python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "20.0_data"
python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "5.0_gaussian_weights"
python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "20.0_gaussian_weights"
python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "20.0_gradient_amplification"
python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "20.0_gradient_inversion"
python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "partial"
python /gpfs/home/griesmax/Federated-Learning/run.py group-data --save-filename "total"