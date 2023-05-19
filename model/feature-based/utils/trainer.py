from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor,MLPClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import os
import pickle
import random

def st_train_wrapper(feat, df, model, transform, input_transform, metric, split_col, label_col):
    #read label columns for later training
    label=df[label_col].to_numpy()

    #apply scaler to improve fitting accuracy
    scaler = StandardScaler()
    if input_transform=='standard':
        print('standard scaler is applied!')
        scaler.fit(feat)
        feat=scaler.transform(feat)
    if transform=='log':
        label=np.log(label)
    if model=='nn-pca':
        pca = PCA(n_components=50,svd_solver='full')
        feat = pca.fit_transform(feat)
    ytrains=np.empty([0,1])
    ytests=np.empty([0,1])

    preds_ytrain=np.empty([0,1])
    preds_ytest=np.empty([0,1])
    metrics=np.empty([0,1])
    r2s=np.empty([0,1])
    model2save=[]

    train_idxs=[]
    valid_idxs=[]
    test_idxs=[]

    split_cols=[col for col in list(df.columns) if split_col in col]
    print('column number:', split_cols)

    if len(split_cols)>0:
        print('start given split')
        for col in split_cols:
            train_index=[i for i, x in enumerate(df[col]=='train') if x]
            valid_index=[i for i, x in enumerate(df[col]=='val') if x]
            test_index=[i for i, x in enumerate(df[col]=='test') if x]

            train_idxs+=[train_index]
            valid_idxs+=[valid_index]
            test_idxs+=[test_index]

            X_train, X_test = feat[train_index+valid_index], feat[test_index]
            y_train, y_test = label[train_index+valid_index], label[test_index]
            
            
            if model=='svm':
                regressor=svm.SVR(gamma='auto')
            elif model=='rf':
                max_depth = 30
                regressor = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=2)
            elif model=='nn'or model=='nn-pca':
                regressor = MLPRegressor(hidden_layer_sizes=[16,16], learning_rate_init=0.001, early_stopping=True, random_state=1, max_iter=2000)
            elif model=='gb-30':
                max_depth = 30
                regressor=GradientBoostingRegressor(random_state=1, max_depth=max_depth)
            elif model=='gb':
                regressor=GradientBoostingRegressor(random_state=1)

            # Train and Predict on ecfp features
            regressor.fit(X_train, y_train)
            y_model = regressor.predict(X_test)
            result_train = regressor.predict(X_train)

            if transform=='log':
                y_test=np.exp(y_test)
                y_train=np.exp(y_train)
                y_model=np.exp(y_model)
                result_train=np.exp(result_train)

            ytests=np.append(ytests, np.atleast_2d(y_test).transpose(), axis=0)
            ytrains=np.append(ytrains, np.atleast_2d(y_train).transpose(), axis=0)
            preds_ytest=np.append(preds_ytest, np.atleast_2d(y_model).transpose(), axis=0)
            preds_ytrain=np.append(preds_ytrain, np.atleast_2d(result_train).transpose(), axis=0)
            if metric=='MAE':
                metrics=np.append(metrics, np.atleast_2d(np.abs(y_model-y_test).mean()), axis=0)
            elif metric=='RMSE':
                metrics=np.append(metrics, np.atleast_2d(np.sqrt(((y_model-y_test)**2).mean())), axis=0)
            r2s=np.append(r2s, np.atleast_2d(r2_score(y_model, y_test)), axis=0)
            model2save+=[regressor]

    else:
        print('start kfold_validation')
        #start 5-fold validation
        k=5
        kf = KFold(n_splits= k, shuffle=True, random_state=1)
        for train_val_index, test_index in kf.split(feat):
            train_index=random.sample(list(train_val_index), int(len(train_val_index)))#*0.75
            valid_index=list(set(train_val_index) - set(train_index))

            train_idxs+=[train_index]
            valid_idxs+=[valid_index]
            test_idxs+=[test_index]

            X_train, X_test = feat[train_index+valid_index], feat[test_index]
            y_train, y_test = label[train_index+valid_index], label[test_index]
            
            if model=='svm':
                regressor=svm.SVR(gamma='auto')
            elif model=='rf':
                max_depth = 30
                regressor = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=2)
            elif model=='nn'or model=='nn-pca':
                regressor = MLPRegressor(hidden_layer_sizes=[16,16], learning_rate_init=0.001, early_stopping=True, random_state=1, max_iter=2000)
            elif model=='gb':
                regressor=GradientBoostingRegressor(random_state=1)

            # Train and Predict on ecfp features
            regressor.fit(X_train, y_train)
            y_model = regressor.predict(X_test)
            result_train = regressor.predict(X_train)

            if transform=='log':
                y_test=np.exp(y_test)
                y_train=np.exp(y_train)
                y_model=np.exp(y_model)
                result_train=np.exp(result_train)

            ytests=np.append(ytests, np.atleast_2d(y_test).transpose(), axis=0)
            ytrains=np.append(ytrains, np.atleast_2d(y_train).transpose(), axis=0)
            preds_ytest=np.append(preds_ytest, np.atleast_2d(y_model).transpose(), axis=0)
            preds_ytrain=np.append(preds_ytrain, np.atleast_2d(result_train).transpose(), axis=0)
            if metric=='MAE':
                metrics=np.append(metrics, np.atleast_2d(np.abs(y_model-y_test).mean()), axis=0)
            elif metric=='RMSE':
                metrics=np.append(metrics, np.atleast_2d(np.sqrt(((y_model-y_test)**2).mean())), axis=0)
            r2s=np.append(r2s, np.atleast_2d(r2_score(y_model, y_test)), axis=0)
            model2save+=[regressor]
    return  metrics, r2s, preds_ytest, ytests, preds_ytrain, ytrains, model2save, train_idxs, valid_idxs, test_idxs


def save_model(model2save, model_dir,idx):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_dir+'model'+str(idx)+'.pkl','wb') as f:
        pickle.dump(model2save,f)

def read_model(model_dir):
    with open(model_dir+'model.pkl', 'rb') as f:
        saved_model=pickle.load(f)
    return saved_model