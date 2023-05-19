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
#SBATCH -e bcf_%j_%a.err
#SBATCH -o bcf_%j_%a.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=shangzhu@andrew.cmu.edu # Email to which notifications will be sent
#SBATCH --array=0,1,2,3,4
echo "Job started on `hostname` at `date`"

#training on cpu nodes
source activate tfgpu

split_prefix='/ocean/projects/cts180021p/shang9/molecules/pretrainLCA/finalized/model/feature-based/result_1/'
data_prefix='/ocean/projects/cts180021p/shang9/molecules/pretrainLCA/finalized/data/o_gnn_input/'
task='BCF/'
python demo.py --folder_idx ${SLURM_ARRAY_TASK_ID} --data_path $data_prefix$task'raw/data.csv' --split_folder $split_prefix$task\
      --dense 1 --dropout 0 --layer 2 

echo " "
echo "Job Ended at `date`"