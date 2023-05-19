#!/bin/bash
#
#SBATCH -J res               # Job name
#SBATCH -n 1                   # Number of total cores
#SBATCH -N 1                    # Number of nodes
#SBATCH --time=0-00:20:00          # Runtime in D-HH:MM
#SBATCH -A cts180021p              # Partition to submit to #gpu
#SBATCH -p RM-shared    # RM-shared
#SBATCH --mem-per-cpu=2000      # Memory pool for all cores in GB (see also --mem-per-cpu
#SBATCH -e feat_clint_%j.err
#SBATCH -o feat_clint_%j.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=shangzhu@andrew.cmu.edu # Email to which notifications will be sent

echo "Job started on `hostname` at `date`"
source activate ognn

data_path='/ocean/projects/cts180021p/shang9/molecules/pretrainLCA/finalized/data/o_gnn_feat_input/'
model_path='/ocean/projects/cts180021p/shang9/molecules/pretrainLCA/finalized/model/o-gnn/result/Clint/model_k4_n0.pt'
for csv_name in 'Clint' 
    do
    python main_get_feature.py --random_seed 15213 \
        --input_path $data_path \
        --input_csv_name $csv_name --gnn 'dualgraph2' \
        --save-test True \
        --batch-size 32 \
        --dropout 0.0 --pooler-dropout 0.0 \
        --init-face --use-bn --num-layers 5 --lr 0.0003 \
        --weight-decay 0.1 --beta2 0.999 --num-workers 1 \
        --mlp-hidden-size 256 --lr-warmup \
        --use-adamw --node-attn --period 25 --checkpoint-dir $model_path
    done

echo " "
echo "Job Ended at `date`"