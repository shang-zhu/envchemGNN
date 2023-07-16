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

Curated datasets can be found at (figshare?): need to check the data policy
1. folder structures:
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

2. Put 'data' folder at 'envchemGNN/data' 

### Train a feature-based model
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

### Train a neuralFP model

### Train a O-GNN model

### Train a O-GNN model

## Acknowledgement

Duvenaud, D. K., et al. (2015). Convolutional Networks on Graphs for Learning Molecular Fingerprints. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, & R. Garnett (Eds.), Advances in Neural Information Processing Systems (Vol. 28).

Zhu, J., et al. (2023). \${\textbackslash}mathcal{O}$-{GNN}: incorporating ring priors into molecular modeling. The Eleventh International Conference on Learning Representations.