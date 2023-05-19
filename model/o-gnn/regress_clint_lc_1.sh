#!/bin/bash
#
#SBATCH -J res               # Job name
#SBATCH -n 1                   # Number of total cores
#SBATCH -N 1                    # Number of nodes
#SBATCH --gpus=v100-32:1
#SBATCH --time=0-4:00:00          # Runtime in D-HH:MM
#SBATCH -A cts180021p              # Partition to submit to #gpu
#SBATCH -p GPU-small    # RM-shared
#SBATCH --mem=32000      # Memory pool for all cores in GB (see also --mem-per-cpu
#SBATCH -e lc_clint_%j.err
#SBATCH -o lc_clint_%j.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=shangzhu@andrew.cmu.edu # Email to which notifications will be sent

echo "Job started on `hostname` at `date`"
source activate ognn

data_path='/ocean/projects/cts180021p/shang9/molecules/pretrainLCA/finalized/data/o_gnn_lc_input/'
split_path='/ocean/projects/cts180021p/shang9/molecules/pretrainLCA/finalized/model/feature-based/result/lc/'
N_epoch=200
#'Clint_1_20' 'Clint_1_10' 
for csv_name in 'Clint_1_5'  
    do
    for st_idx in 0 1 2 3 4
        do
        python main_dg_regress.py --random_seed 15213 \
            --input_path $data_path \
            --input_csv_name $csv_name --gnn 'dualgraph2' \
            --save-test True \
            --batch-size 32 \
            --dropout 0.0 --pooler-dropout 0.0 \
            --init-face --use-bn --epochs $N_epoch --num-layers 5 --lr 0.0003 \
            --weight-decay 0.1 --beta2 0.999 --num-workers 1 \
            --mlp-hidden-size 256 --lr-warmup \
            --use-adamw --node-attn --period 25 \
            --split_folder $split_path$csv_name'/' \
            --kfold_idx $st_idx
        done
    done
echo " "
echo "Job Ended at `date`"