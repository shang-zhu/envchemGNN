import deepchem as dc
import numpy as np
import sys, os
import argparse
import random
import pandas as pd
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe, Trials
import tempfile
import shutil

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

def fm(args):
    rmses=[]
    for seed in range(5):
        try:
            if folder_path!='':
                    train_idx, val_idx, test_idx=get_split_idx_from_file(folder_path, seed)
            else:
                    df=pd.read_csv(data_path)
                    train_idx, val_idx, test_idx=get_split_idx(df.count()[0])

            # print('training-validation-testing split index:', train_idx, val_idx, test_idx)
            loader = dc.data.CSVLoader(['label'], feature_field="SMILES",
                    featurizer=dc.feat.ConvMolFeaturizer())
            data = loader.create_dataset(data_path)
            splitter = dc.splits.SpecifiedSplitter(valid_indices=val_idx, test_indices=test_idx)
            train_dataset, val_dataset, test_dataset =splitter.train_valid_test_split(data)
            metrics_key='rms_score'
            metrics=dc.metrics.Metric(dc.metrics.rms_score, mode="regression")

            save_dir ='./temp/'+data_path.split('/')[-3]+'_'+str(seed)+'/'
            
            model = dc.models.GraphConvModel(n_tasks=1, graph_conv_layers=args['layers'],\
                    dense_layer_size=args['dense'],dropout=args['dropout'], mode='regression')
            #validation callback that saves the best checkpoint, i.e the one with the maximum score.
            validation=dc.models.ValidationCallback(test_dataset, 10, [metrics],save_dir=save_dir,save_on_minimum=True)
            
            model.fit(train_dataset, nb_epoch=100,callbacks=validation)


            #restoring the best checkpoint and passing the negative of its validation score to be minimized.
            model.restore(model_dir=save_dir)
            valid_score = model.evaluate(test_dataset, [metrics])
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)

            rmses+=[valid_score[metrics_key]]
        except:
            pass

    return np.array(rmses).mean() 


parser = argparse.ArgumentParser(description="DeepChem Baseline Models for green chem")
# parser.add_argument("--folder_idx", type=int, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--split_folder", type=str, required=True)


search_space = {
    'layers': hp.choice('layers',[[64, 64], [128,128], [64,64,64]]),
    'dense': hp.choice('dense',[64, 128]),
    'dropout': hp.choice('dropout',[0, 0.2]),

}

args = parser.parse_args()
print(args)

# seed=args.folder_id
data_path=args.data_path
folder_path=args.split_folder

reproduce(15213)

try:
    trials=Trials()
    best = fmin(fm,
                space= search_space,
                algo=tpe.suggest,
                max_evals=15,
                trials = trials)

    print(data_path.split('/')[-3],"Best: {}".format(best))
except:
    pass