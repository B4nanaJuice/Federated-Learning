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

python run.py group-data --run-count 10 --save-filename "clean_run"
python run.py group-data --run-count 10 --save-filename "5%_clients_data"
python run.py group-data --run-count 10 --save-filename "20%_clients_data"
python run.py group-data --run-count 10 --save-filename "5%_clients_model"
python run.py group-data --run-count 10 --save-filename "20%_clients_model"
python run.py group-data --run-count 10 --save-filename "clients_gradient_inversion"
python run.py group-data --run-count 10 --save-filename "clients_gradient_amplification"