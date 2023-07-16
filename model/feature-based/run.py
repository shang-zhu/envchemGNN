import numpy as np
import os
import pandas as pd
from utils import trainer
import argparse

#loading parameters
parser = argparse.ArgumentParser()
parser.add_argument('--label_path', type=str, required=True)
parser.add_argument('--label_name', type=str, required=True)
parser.add_argument('--feat_path', type=str, required=True)
parser.add_argument('--result_path', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--metric', type=str, required=True)
parser.add_argument('--split_col', type=str, required=False, default='None')
parser.add_argument('--save_model', action='store_true')
parser.set_defaults(save_model=False)

args = parser.parse_args()
label_path=args.label_path
label=args.label_name
feat_path=args.feat_path
task=args.task
metric=args.metric
split_col=args.split_col
save_model=args.save_model
save_model_name=label_path.split('/')[-1].split('.')[0]
result_path=args.result_path

#start this regression task
print('this task is on regression of', save_model_name)

#loading features
feat_list=['ecfp', 'maccs', 'mordred']
feat_arrs=[]
for feat in feat_list:
    npy_path=feat_path+feat+'/feat.npy'
    if os.path.exists(npy_path):
        feat_arrs+=[np.load(npy_path)]
        print('loading '+ feat +' features successfully!')

#loading labels from csv
df=pd.read_csv(label_path)

print('loading dataframe successfully!')

#set the models to be tested
model_list=['rf', 'svm', 'gb', 'nn', 'nn-pca', 'gb-30'] #

#creating results folder
if not os.path.exists(result_path+save_model_name+'/'):
    os.makedirs(result_path+save_model_name+'/')
    print("The result directory is created!")

col = {'Model':[], 'Feat': [], 'Label':[], metric+'_mean':[], metric+'_std':[]}
model_perf_df=pd.DataFrame(col)
for feat_idx, feat in enumerate(feat_list):
    for model_idx, model in enumerate(model_list): 
        #results, we use k-fold validation k=5
        loss,  _, result, ytest, result_train, ytrain, model2save, train_idxs, valid_idxs, test_idxs =\
            trainer.st_train_wrapper(feat_arrs[feat_idx], df, model, \
                transform='raw', input_transform='raw', metric=metric, split_col=split_col, label_col=label)

        #outputting results
        print('Regression on %s, Feature - %s, Model - %s :' \
            %(label, feat, model))
        print(metric+' %0.2E +/- %0.2E' \
            %(loss.mean(axis=0), loss.std(axis=0)))
        model_perf_df.loc[len(model_perf_df.index)] = \
            [model, feat, label, loss.mean(axis=0)[0], loss.std(axis=0)[0]]
        
        #saving models
        if save_model:
            for idx in range(len(model2save)):
                model_dir=result_path+save_model_name+'/saved_model/'+feat+'/'+model+'/'
                trainer.save_model(model2save[idx], model_dir, idx)
                    
#saving training-validation-testing splits to test other models               
for fold, train_idx in enumerate(train_idxs):
    with open(result_path+save_model_name+'/train_idx_'+str(fold)+'.txt', 'a+') as fp:
        for idx in train_idx:
            fp.write("%s\n" % idx)

for fold, valid_idx in enumerate(valid_idxs):
    with open(result_path+save_model_name+'/valid_idx_'+str(fold)+'.txt', 'a+') as fp:
        for idx in valid_idx:
            fp.write("%s\n" % idx)

for fold, test_idx in enumerate(test_idxs):
    with open(result_path+save_model_name+'/test_idx_'+str(fold)+'.txt', 'a+') as fp:
        for idx in test_idx:
            fp.write("%s\n" % idx)
            
#summary of prediction results
model_perf_df.to_csv(result_path+save_model_name+'/summary_kfold.csv', index=False)
    

