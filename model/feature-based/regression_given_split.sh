#!/bin/bash
#
#SBATCH -J res               # Job name
#SBATCH -n 1                   # Number of total cores
#SBATCH -N 1                    # Number of nodes
#SBATCH --gpus=v100-32:1
#SBATCH --time=0-1:00:00          # Runtime in D-HH:MM
#SBATCH -A cts180021p              # Partition to submit to #gpu
#SBATCH -p GPU-shared    # RM-shared
#SBATCH --mem=32000      # Memory pool for all cores in GB (see also --mem-per-cpu
#SBATCH -e given_split_%j.err
#SBATCH -o given_split_%j.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=shangzhu@andrew.cmu.edu # Email to which notifications will be sent

echo "Job started on `hostname` at `date`"

source activate ml_env

data_path='/ocean/projects/cts180021p/shang9/molecules/pretrainLCA/finalized/data/'
#  
for csv_name in 'SO4_react' 'O3_react' 'fup' 'OH_react'
    do
    # python data.py --input_path $data_path'model_input/given_split/'$csv_name'.csv' \
    #     --output_path $data_path'features/'$csv_name'/'

    python run.py --feat_path $data_path'features/'$csv_name'/' \
        --label_path $data_path'model_input/given_split/'$csv_name'.csv' --label_name 'label' \
        --task 'regression' --metric 'RMSE' --split_col 'set' --save_model --result_path 'result/'
    done
    
echo " "
echo "Job Ended at `date`"