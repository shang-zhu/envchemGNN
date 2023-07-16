# envchemGNN

This is the official repository for manuscript 'Improved Environmental Chemistry Property Prediction of Molecules with Graph Machine Learning'.

The implementation of NeuralFP is from [DeepChem](https://deepchem.io/) and the implementation of OGNN is from [this repo](https://github.com/O-GNN/O-GNN).

## Getting Started

### Installation

```
#create conda environment
conda create --name ml_echem python=3.7
conda activate ml_echem

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
        --split_folder $result_path$task'NeuralFP/' --dense 1 --dropout 0 --layer 1
```
2. prediction results are saved at (5 files from an ensemble of 5 models): 
```$result_path$task'NeuralFP/test_pred_'+str(split_id)+'_'+[0-4]+'.npy'```
with true labels
```$result_path$task'NeuralFP/test_'+str(split_id)+'.npy'```.
Then you can average the prediction of model ensembles and compare the predictions with true labels, get RMSE or other metrics.

### Train a O-GNN model

## Acknowledgement

Duvenaud, D. K., et al. (2015). Convolutional Networks on Graphs for Learning Molecular Fingerprints. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, & R. Garnett (Eds.), Advances in Neural Information Processing Systems (Vol. 28).

Zhu, J., et al. (2023). \${\textbackslash}mathcal{O}$-{GNN}: incorporating ring priors into molecular modeling. The Eleventh International Conference on Learning Representations.