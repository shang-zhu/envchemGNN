# envchemGNN

This is the official repository for manuscript 'Improved Environmental Chemistry Property Prediction of Molecules with Graph Machine Learning'.

The implementation of NeuralFP is from deepchem and the implementation of OGNN is from [this repo](https://github.com/O-GNN/O-GNN).


## Getting Started

### Installation

```
#create conda environment
conda create --name ml_echem python=3.7
conda activate ml_echem

#install packages for model training
pip install -U scikit-learn 
conda install -c rdkit -c mordred-descriptor mordred
conda install pyg -c pyg
pip install deepchem[tensorflow]

#other packages for data analysis
pip install pandas matplotlib
```

### Dataset

1. you should put 'feature_result' folder as 'envchemGNN/model/feature-based/result_1', while 'feature_result_stand_scaler' folder as 'envchemGNN/model/feature-based/result_stand_scaler_1'

2. you should put 'data' folder as 'envchemGNN/data'

3. you should put 'deepchem_result' folder as 'envchemGNN/model/deepchem/publish/result'

4. you should put 'ognn_result' folder as 'envchemGNN/model/o-gnn/result', while 'ognn_graph_features' folder as 'envchemGNN/model/o-gnn/graph_features'

### Train a feature-based model

### Train a neuralFP model

### Train a O-GNN model

### Train a O-GNN model