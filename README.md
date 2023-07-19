# envchemGNN

This is the official repository for manuscript 'Improved Environmental Chemistry Property Prediction of Molecules with Graph Machine Learning'.

The implementation of NeuralFP is from [DeepChem](https://deepchem.io/) and the implementation of OGNN is from [this repo](https://github.com/O-GNN/O-GNN).

## Getting Started

### Installation
First, clone the github repository:

```
git clone https://github.com/shangzhu-cmu/envchemGNN.git
cd envchemGNN
```

You can then install packages by creating an environment with the provided yaml file

```
conda env create -f environment.yml
```

Alternatively, you can install pakcages manually with the following instructions

```
#create conda environment
conda create --name ecgnn python=3.7
conda activate ecgnn

#install packages for model training
#ognn
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install pytorch-scatter pytorch-sparse -c pyg -c pytorch -c nvidia -c conda-forge
pip install ogb==1.3.5
pip install tensorboard
conda install -c rdkit rdkit
#feature-based, deepchem
pip install -U scikit-learn 
conda install -c rdkit -c mordred-descriptor mordred
pip install --pre deepchem[tensorflow]

#other packages for data analysis
pip install pandas matplotlib
```

### Dataset

Curated datasets can be found at './data': need to check the data policy
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
-o_gnn_input: input files for ognn model
-lc
--'BCF_1_N.csv': 1/N randomly sampled data from 'BCF.csv' for learning curve
--'Clint_1_N.csv': 1/N randomly sampled data from 'Clint.csv' for learning curve
``` 

### Train a feature-based model

1. run the following commands in your terminal
```
cd ./model/feature-based
data_path='./data/'
result_path='./result/feature_based/'
task='ESOL' # other tasks: 'BCF' 'Clint'

#create features
python data.py --input_path $data_path'random_split/'$task'.csv' \
        --output_path $data_path'features/'$task'/'

#train models (random split)
python run.py --feat_path $data_path'features/'$task'/' \
    --label_path $data_path'random_split/'$task'.csv' --label_name 'label' \
    --task 'regression' --metric 'RMSE' --save_model --result_path $result_path
```

2. The testing results are saved at ```$result_path/$csv_name/summary_kfold.csv``` for further analysis, along with the training-validation-testing-split indexes, and saved models. Notice that validation set is not necessary for the feature-based models except for Neural-networks, the validation-idx is empty.

3. We observed a small stochasticity of model training, which didn't influence the model selection results.


### Train a neuralFP model

1. run the following commands in your terminal
```
cd ./deepchem/
result_path='./result/'
data_path='./data/'
task='ESOL'
split_id=0 # will run id=1,2,3,4 for cross validation

#need to first run feature-based models above, so that train-valid-test splits are provided. otherwise remove '--split_folder'.
python run.py --folder_idx $split_id --data_path $data_path'random_split/'$task'.csv' --split_folder $result_path'feature_based/'$task'/' --result_path $result_path'neuralFP/'$task'/' \
        --dense 1 --dropout 0 --layer 1
```
2. prediction results are saved at (5 files from an ensemble of 5 models): 
```$result_path'neuralFP/'$task'NeuralFP/preds_'+str(split_id)+'.csv'```
which has 7 columns, including idx(test set indexs from original dataset), y_true (true labels) and y_pred_[0-4] from an esemble of 5 models.
Then you can average the prediction of model ensembles and compare the predictions with true labels, get RMSE or other metrics. A small stochasticity was observed during model training, which didn't influence the qualitative conclusions.

3. optimized hyperparameters for other tasks:

```
'BCF': --dense 1 --dropout 0 --layer 2
'Clint': --dense 1 --dropout 0 --layer 2
'O3': --dense 1 --dropout 0 --layer 0
'SO4': --dense 1 --dropout 0 --layer 2
```

### Train a O-GNN model
1. run the following command in your terminal
```
cd ./model/o-gnn
result_path='./result/'
data_path='./data/o_gnn_input/'
task='ESOL'
split_id=0 # will run id=1,2,3,4 for cross validation
N_epoch=200 #400 for BCF and SO4

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

```
2.  prediction results are saved at : 
```$result_path'o-gnn/'$task'/preds_'+str(split_id)+'.csv'```
which has 7 columns, including idx(test set indexs from original dataset), y_true (true labels) and y_pred_[0-4] from an esemble of 5 models.
Then you can average the prediction of model ensembles and compare the predictions with true labels, get RMSE or other metrics. A small stochasticity was observed during model training, which didn't influence the qualitative conclusions.

## Acknowledgement

### Model Implementations:

Duvenaud, D. K., et al. (2015). Convolutional Networks on Graphs for Learning Molecular Fingerprints. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, & R. Garnett (Eds.), Advances in Neural Information Processing Systems (Vol. 28).

Zhu, J., et al. (2023). \${\textbackslash}mathcal{O}$-{GNN}: incorporating ring priors into molecular modeling. The Eleventh International Conference on Learning Representations.

### Data Sources:

ESOL: Delaney, J. S. (2004). Journal of Chemical Information and Computer Sciences, 44(3), 1000–1005. [link](https://doi.org/10.1021/ci034243x)
BCF: Grisoni, F., et al. (2015). Chemosphere, 127, 171–179.  [link](https://doi.org/https://doi.org/10.1016/j.chemosphere.2015.01.047)
Clint: Dawson, D. E., et al. (2021). Environmental Science & Technology, 55(9), 6505–6517. [link](https://doi.org/10.1021/acs.est.0c06117)
O3/SO4: Zhong, S., et al. (2022). Environmental Science & Technology, 56(1), 681–692. [link](https://doi.org/10.1021/acs.est.1c04883)
