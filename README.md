# envchemGNN

This is the repository for manuscript 'Improved Environmental Chemistry Property Prediction of Molecules with Graph Machine Learning'. full code will be coming soon.

The implementation of NeuralFP is from deepchem and the implementation of OGNN is from [this repo](https://github.com/O-GNN/O-GNN).

## To install all necessary packages

```
conda create --name ml_echem python=3.7
conda activate ml_echem
pip install matplotlib
pip install -U scikit-learn 
conda install -c rdkit -c mordred-descriptor mordred

#deepchem for neuralFP
pip install deepchem[tensorflow]
pip install deepchem[torch]

#ognn
conda install pyg -c pyg

#for data analysis
pip install pandas
```




## To obtain the data, do the following:

1. you should put 'feature_result' folder as 'envchemGNN/model/feature-based/result_1', while 'feature_result_stand_scaler' folder as 'envchemGNN/model/feature-based/result_stand_scaler_1'

2. you should put 'data' folder as 'envchemGNN/data'

3. you should put 'deepchem_result' folder as 'envchemGNN/model/deepchem/publish/result'

4. you should put 'ognn_result' folder as 'envchemGNN/model/o-gnn/result', while 'ognn_graph_features' folder as 'envchemGNN/model/o-gnn/graph_features'

