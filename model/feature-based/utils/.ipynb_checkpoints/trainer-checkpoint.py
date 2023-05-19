import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import os
import pickle
#[123, 234, 345, 456, 567]

def st_train_wrapper(feat, label, train_ratio, model, transform, input_transform, sds, cross_valid='k-fold'):
    scaler = StandardScaler()
    if input_transform=='standard':
        scaler.fit(feat)
        feat=scaler.transform(feat)
# if normalizer=='minmax':
#     scaler = MinMaxScaler()
# elif normalizer=='standard':
#     scaler = StandardScaler()

    if transform=='log':
        label=np.log(label)
    if model=='nn-pca':
        pca = PCA(n_components=50,svd_solver='full')
        feat = pca.fit_transform(feat)
    train_dim=int(feat.shape[0]*train_ratio)
    test_dim=int(feat.shape[0]*(1-train_ratio))+1
    ytrains=np.empty([0,1])
    ytests=np.empty([0,1])

    results_train=np.empty([0,1])
    results=np.empty([0,1])
    maes=np.empty([0,1])
    r2s=np.empty([0,1])
    model2save=[]

    if cross_valid=='k-fold':
        k=5
        kf = KFold(n_splits= k, shuffle=True, random_state=1)
        for train_index, test_index in kf.split(feat):
            X_train_ecfp, X_test_ecfp = feat[train_index], feat[test_index]
            y_train, y_test = label[train_index], label[test_index]
            
            if model=='svm':
                rf=svm.SVR(gamma='auto')
            elif model=='rf':
                max_depth = 30
                rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=2)
            elif model=='nn'or model=='nn-pca':
                rf = MLPRegressor(hidden_layer_sizes=[16,16], learning_rate_init=0.001, early_stopping=True, random_state=1, max_iter=2000)
            elif model=='gb':
                rf=GradientBoostingRegressor(random_state=1)

            # Train and Predict on ecfp features
            rf.fit(X_train_ecfp, y_train)
            y_rf_ecfp = rf.predict(X_test_ecfp)
            result_train = rf.predict(X_train_ecfp)

            if transform=='log':
                y_test=np.exp(y_test)
                y_train=np.exp(y_train)
                y_rf_ecfp=np.exp(y_rf_ecfp)
                result_train=np.exp(result_train)

            ytests=np.append(ytests, np.atleast_2d(y_test).transpose(), axis=0)
            ytrains=np.append(ytrains, np.atleast_2d(y_train).transpose(), axis=0)
            results=np.append(results, np.atleast_2d(y_rf_ecfp).transpose(), axis=0)
            results_train=np.append(results_train, np.atleast_2d(result_train).transpose(), axis=0)
            maes=np.append(maes, np.atleast_2d(np.abs(y_rf_ecfp-y_test).mean()), axis=0)
            r2s=np.append(r2s, np.atleast_2d(r2_score(y_rf_ecfp, y_test)), axis=0)
            model2save+=[rf]
    
    elif cross_valid=='random':   
        for sd in sds:
            X_train_ecfp, X_test_ecfp, y_train, y_test = train_test_split(
                    feat, label, train_size=train_ratio, random_state=sd)

            if model=='svm':
                rf=svm.SVR(gamma='auto')
            elif model=='rf':
                max_depth = 30
                rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=2)
            elif model=='nn'or model=='nn-pca':
                rf = MLPRegressor(hidden_layer_sizes=[16,16], learning_rate_init=0.001, early_stopping=True, random_state=1, max_iter=3000)
            elif model=='gb':
                rf=GradientBoostingRegressor(random_state=1)

            # Train and Predict on ecfp features
            rf.fit(X_train_ecfp, y_train)
            y_rf_ecfp = rf.predict(X_test_ecfp)
            result_train = rf.predict(X_train_ecfp)

            if transform=='log':
                y_test=np.exp(y_test)
                y_train=np.exp(y_train)
                y_rf_ecfp=np.exp(y_rf_ecfp)
                result_train=np.exp(result_train)

            ytests=np.append(ytests, np.atleast_2d(y_test).transpose(), axis=0)
            ytrains=np.append(ytrains, np.atleast_2d(y_train).transpose(), axis=0)
            results=np.append(results, np.atleast_2d(y_rf_ecfp).transpose(), axis=0)
            results_train=np.append(results_train, np.atleast_2d(result_train).transpose(), axis=0)
            maes=np.append(maes, np.atleast_2d(np.abs(y_rf_ecfp-y_test).mean()), axis=0)
            r2s=np.append(r2s, np.atleast_2d(r2_score(y_rf_ecfp, y_test)), axis=0)
            model2save+=[rf]
    return  maes, r2s, results, ytests, results_train, ytrains, model2save

def save_model(model2save, model_dir,idx):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_dir+'model'+str(idx)+'.pkl','wb') as f:
        pickle.dump(model2save,f)

def read_model(model_dir):
    with open(model_dir+'model.pkl', 'rb') as f:
        saved_model=pickle.load(f)
    return saved_model