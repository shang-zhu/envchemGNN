#!/bin/bash
#
#SBATCH -J res               # Job name
#SBATCH -n 1                   # Number of total cores
#SBATCH -N 1                    # Number of nodes
##SBATCH --gpus=v100-32:1
#SBATCH --time=0-8:00:00          # Runtime in D-HH:MM
#SBATCH -A cts180021p              # Partition to submit to #gpu
#SBATCH -p RM-shared
#SBATCH --mem-per-cpu=2000      # Memory pool for all cores in GB (see also --mem-per-cpu
#SBATCH -e o3_%j.err
#SBATCH -o o3_%j.out # File to which STDOUT will be written %j is the job #

echo "Job started on `hostname` at `date`"

#training on cpu nodes
source activate ml_echem

split_prefix='/ocean/projects/cts180021p/shang9/molecules/pretrainLCA/finalized/model/feature-based/result_1/'
data_prefix='/ocean/projects/cts180021p/shang9/molecules/pretrainLCA/finalized/data/o_gnn_input/'
task='O3_react/'
python hyper.py --data_path $data_prefix$task'raw/data.csv' --split_folder $split_prefix$task

echo " "
echo "Job Ended at `date`"