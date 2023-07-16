# envchemGNN

This is the official repository for manuscript 'Improved Environmental Chemistry Property Prediction of Molecules with Graph Machine Learning'.

The implementation of NeuralFP is from [DeepChem](https://deepchem.io/) and the implementation of OGNN is from [this repo](https://github.com/O-GNN/O-GNN).

## Getting Started

### Installation

```
#create conda environment
conda create --name ecgnn python=3.7
conda activate ecgnn

#install packages for model training
pip install -U scikit-learn 
conda install -c rdkit -c mordred-descriptor mordred

#other packages for data analysis
pip install pandas matplotlib

#clone this folder
git clone https://github.com/shangzhu-cmu/envchemGNN.git
cd envchemGNN
```

### Dataset

Curated datasets can be found at 'envchemGNN/data': need to check the data policy
folder structures:
```
data
-random_split
--'BCF.csv': Bioconcentration
--'Clint.csv': Intrinsic Clearance
--'ESOL.csv': Solubility
-given_split
--'O3_react.csv': Reactivity
--'SO4_react.csv': Reactivity
-lc
--'BCF_1_N.csv': 1/N randomly sampled data from 'BCF.csv' for learning curve
--'Clint_1_N.csv': 1/N randomly sampled data from 'Clint.csv' for learning curve
``` 

### Train a feature-based model

1. run the following commands in your terminal
```
data_path='your_local_dir/envchemGNN/data/'
result_path='your_local_dir/envchemGNN/result/'
csv_name='ESOL' # other tasks: 'BCF' 'Clint'

#create features
python data.py --input_path $data_path'model_input/random_split/'$csv_name'.csv' \
        --output_path $data_path'features/'$csv_name'/'

#train models (random split)
python run.py --feat_path $data_path'features/'$csv_name'/' \
    --label_path $data_path'model_input/random_split/'$csv_name'.csv' --label_name 'label' \
    --task 'regression' --metric 'RMSE' --save_model --result_path $result_path
```

2. The testing results are saved at $result_path/feature_result/$csv_name/summary_kfold.csv for further analysis

### Train a neuralFP model

1. run the following commands in your terminal
```
result_path='your_local_dir/envchemGNN/result/'
data_path='your_local_dir/envchemGNN/data/'
task='ESOL'
split_id=0 # will run id=1,2,3,4 for cross validation

python run.py --folder_idx $split_id --data_path $data_path'model_input/random_split/'$task'.csv' 
        --split_folder $result_path$task --dense 1 --dropout 0 --layer 1
```
2. prediction results are saved at (5 files from an ensemble of 5 models): 
```$result_path$task'NeuralFP/test_pred_'+str(split_id)+'_'+[0-4]+'.npy'```
with true labels
```$result_path$task'NeuralFP/test_'+str(split_id)+'.npy'```.
Then you can average the prediction of model ensembles and compare the predictions with true labels, get RMSE or other metrics.

3. optimized hyperparameters for other tasks:

### Train a O-GNN model
1. run the following command in your terminal
```
result_path='your_local_dir/envchemGNN/result/'
data_path='your_local_dir/envchemGNN/data/o_gnn_input/'
task='Clint'
split_id=0 # will run id=1,2,3,4 for cross validation
N_epoch=200 #400 for BCF and SO4

python main_dg_regress.py --random_seed 15213 \
        --input_path $data_path \
        --input_csv_name $task --gnn 'dualgraph2' \
        --save-test True \
        --batch-size 32 \
        --dropout 0.0 --pooler-dropout 0.0 \
        --init-face --use-bn --epochs $N_epoch --num-layers 5 --lr 0.0003 \
        --weight-decay 0.1 --beta2 0.999 --num-workers 1 \
        --mlp-hidden-size 256 --lr-warmup \
        --use-adamw --node-attn --period 25 \
        --split_folder $result_path$task \
        --kfold_idx $split_id
```
2.  prediction results are saved at : 
```$result_path$task'OGNN/preds_'+str(split_id)+'.csv'```
which has 6 columns, including y_true (true labels) and y_pred_[0-4] from an esemble of 5 models.
Then you can average the prediction of model ensembles and compare the predictions with true labels, get RMSE or other metrics.

3. Extract the features after graph pooling layer:
```
result_path='your_local_dir/envchemGNN/result/'
data_path='your_local_dir/envchemGNN/data/o_gnn_input/'
task='Clint'
model_path=$result_path$task'OGNN/model_k4_n0.pt'

#in the get_feature.py we set ddi=True in order to extract graph features
python get_feature.py --random_seed 15213 \
        --input_path $data_path \
        --input_csv_name $task --gnn 'dualgraph2' \
        --save-test True \
        --batch-size 32 \
        --dropout 0.0 --pooler-dropout 0.0 \
        --init-face --use-bn --num-layers 5 --lr 0.0003 \
        --weight-decay 0.1 --beta2 0.999 --num-workers 1 \
        --mlp-hidden-size 256 --lr-warmup \
        --use-adamw --node-attn --period 25 --checkpoint-dir $model_path
```

## Acknowledgement

Duvenaud, D. K., et al. (2015). Convolutional Networks on Graphs for Learning Molecular Fingerprints. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, & R. Garnett (Eds.), Advances in Neural Information Processing Systems (Vol. 28).

Zhu, J., et al. (2023). \${\textbackslash}mathcal{O}$-{GNN}: incorporating ring priors into molecular modeling. The Eleventh International Conference on Learning Representations.
