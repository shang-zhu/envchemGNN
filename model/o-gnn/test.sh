#!/bin/bash
#
#SBATCH -J res               # Job name
#SBATCH -n 1                   # Number of total cores
#SBATCH -N 1                    # Number of nodes
#SBATCH --gpus=v100-32:1
#SBATCH --time=0-8:00:00          # Runtime in D-HH:MM
#SBATCH -A cts180021p              # Partition to submit to #gpu
#SBATCH -p GPU-small    # RM-shared
#SBATCH --mem=32000      # Memory pool for all cores in GB (see also --mem-per-cpu
#SBATCH -e regress_esol_%j.err
#SBATCH -o regress_esol_%j.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=shangzhu@andrew.cmu.edu # Email to which notifications will be sent

echo "Job started on `hostname` at `date`"

your_local_dir='/ocean/projects/cts180021p/shang9/published_repos/test'
result_path=$your_local_dir'/envchemGNN/result/'
data_path=$your_local_dir'/envchemGNN/data/o_gnn_input/'
task='ESOL'
# split_id=0 # will run id=1,2,3,4 for cross validation
N_epoch=200 #400 for BCF and SO4
source activate ecgnn2
#load cuda module in slurm
for split_id in 1 2 3 4
        do
                python run.py --random_seed 15213 \
                        --input_path $data_path \
                        --input_csv_name $task --gnn 'dualgraph2' \
                        --save-test True \
                        --batch-size 32 \
                        --dropout 0.0 --pooler-dropout 0.0 \
                        --init-face --use-bn --epochs $N_epoch --num-layers 5 --lr 0.0003 \
                        --weight-decay 0.1 --beta2 0.999 --num-workers 1 \
                        --mlp-hidden-size 256 --lr-warmup \
                        --use-adamw --node-attn --period 25 \
                        --split_folder $result_path'feature_based/'$task'/' \
                        --kfold_idx $split_id \
                        --result_path $result_path'o-gnn/'$task'/'
        done

echo " "
echo "Job Ended at `date`"