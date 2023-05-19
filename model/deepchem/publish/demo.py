import deepchem as dc
import numpy as np
import sys, os
import argparse
import random
import pandas as pd
from sklearn.model_selection import KFold

def reproduce(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_split_idx(size):
    train_idx_list=random.sample(range(size), int(size*0.8))
    other_idx_list=list(set(list(range(size))) - set(train_idx_list))
    val_idx_list=random.sample(other_idx_list, int(size*0.1))
    test_idx_list=list(set(other_idx_list) - set(val_idx_list))
    return train_idx_list, val_idx_list, test_idx_list

def txt2list(txtfile):
    file = open(txtfile, "r")
    data = file.read()
    set_idx = data.split("\n")
    set_idx=[int(idx) for idx in set_idx if idx!='']
    file.close()
    return list(set(set_idx))

def get_split_idx_from_file(split_folder, kfold_idx):
    train_txt=split_folder+'train_idx_'+str(kfold_idx)+'.txt'
    valid_txt=split_folder+'valid_idx_'+str(kfold_idx)+'.txt'
    test_txt=split_folder+'test_idx_'+str(kfold_idx)+'.txt'

    train_idx=txt2list(train_txt)
    valid_idx=txt2list(valid_txt)
    test_idx=txt2list(test_txt)
    return train_idx, valid_idx, test_idx

parser = argparse.ArgumentParser(description="DeepChem Baseline Models for green chem")
parser.add_argument("--folder_idx", type=int, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--split_folder", type=str, required=True)
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--dense", type=int, required=True)
parser.add_argument("--dropout", type=int, required=True)

args = parser.parse_args()
print(args)

seed=args.folder_idx
data_path=args.data_path
folder_path=args.split_folder
save_folder='result/'+folder_path.split('/')[-2]+'/'
layer_id=args.layer
dense_id=args.dense
dropout_id=args.dropout

hyperpara_space = {
    'layer':[[64, 64], [128,128], [64,64,64]],
    'dense':[64, 128],
    'dropout':[0, 0.2],
} 

layer=hyperpara_space['layer'][layer_id]
dense=hyperpara_space['dense'][dense_id]
dropout=hyperpara_space['dropout'][dropout_id]

if not os.path.exists(save_folder):
   # Create a new directory because it does not exist
   try:
       os.makedirs(save_folder)
   except:
       pass
reproduce(seed)

if args.split_folder!='':
        full_train_idx, _, test_idx=get_split_idx_from_file(folder_path, seed)
else:
        df=pd.read_csv(data_path)
        train_idx, val_idx, test_idx=get_split_idx(df.count()[0])

k=5
kf = KFold(n_splits= k, shuffle=True, random_state=1)

k_idx=0
for train_idx, val_idx in kf.split(full_train_idx):
    print('kfold index:', k_idx)
    # print('training-validation-testing split index:', train_idx, val_idx, test_idx)
    loader = dc.data.CSVLoader(['label'], feature_field="SMILES",
            featurizer=dc.feat.ConvMolFeaturizer())
    data = loader.create_dataset(data_path)
    splitter = dc.splits.SpecifiedSplitter(valid_indices=val_idx, test_indices=test_idx)
    train_data, val_data, test_data =splitter.train_valid_test_split(data)
    np.save(save_folder+'test_'+str(seed)+'.npy', test_data.y)

    # print(train_data, val_data, test_data)

    n_tasks = 1
    nepochs=100

    model = dc.models.GraphConvModel(n_tasks,graph_conv_layers=layer,\
                    dense_layer_size=dense,dropout=dropout, mode='regression')
    
    metrics_key='rms_score'
    metrics=dc.metrics.Metric(dc.metrics.rms_score, mode="regression")
    freq=10

    val_opt=float('inf')
    test_opt=float('inf')
    for i_output in range(nepochs):
        
        model.fit(train_data, nb_epoch=freq)
        train_loss=model.evaluate(train_data, metrics)[metrics_key]
        val_loss=model.evaluate(val_data, metrics)[metrics_key]
        test_loss=model.evaluate(test_data, metrics)[metrics_key]
        if val_loss<val_opt:
            val_opt=val_loss
            test_opt=test_loss
            predict_test=model.predict(test_data)
            np.save(save_folder+'test_pred_'+str(seed)+'_'+str(k_idx)+'.npy', predict_test)
        
        print('Epoch', (i_output+1)*freq, ': train ', train_loss,
        ', valid ', val_loss, ', test ',  test_loss,'\n')

    print('best valid loss:', val_opt)
    print('best test loss:', test_opt)
    k_idx+=1