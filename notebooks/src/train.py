import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .train_ann import Dataset, ANN, train_ann

def train(X_train, X_test, y_train, y_test, model):
    if model == 'LightGBM':
        train_data = lgb.Dataset(X_train, label=y_train.astype(float), categorical_feature='IGBP')
        test_data = lgb.Dataset(X_test, label=y_test.astype(float), reference=train_data, categorical_feature='IGBP')

        best_params = {'num_leaves': 80,
         'learning_rate': 0.05,
         'objective': 'regression',
         'metric': 'rmse',
         'boosting_type': 'gbdt',
         'verbose': -1,
         'num_threads': -1,
         'colsample_bytree': 0.7}

        lgbm = lgb.train(best_params, train_data, valid_sets=[test_data], num_boost_round=5000,
                        callbacks=[lgb.early_stopping(stopping_rounds=50)])
        y_pred = lgbm.predict(X_test)
    
    elif model == 'XGBoost':
        X_train['IGBP'] = X_train.IGBP.astype('category')
        X_test['IGBP'] = X_test.IGBP.astype('category')
        best_params = {'learning_rate': 0.1, 'max_depth': 14, 'min_child_weight': 5, 'gamma': 0.6, 'n_estimators': 100, 
               'colsample_bytree': 0.9, 'subsample': 0.8, 'objective': 'reg:squarederror', 'random_state': 42, 
               'verbosity': 0, 'tree_method': 'gpu_hist', 'n_jobs': -1, 'enable_categorical': True}
        model =  xgb.XGBRegressor(**best_params, device="cuda")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=0
        )
        y_pred = model.predict(X_test)
        
    elif model == 'RF':
        best_params = {'max_depth': 22, 'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        
        X_train['IGBP'] = X_train.IGBP.astype('category')
        X_test['IGBP'] = X_test.IGBP.astype('category')

        one_hot = pd.get_dummies(X_train.IGBP)
        X_train = pd.concat([X_train.drop('IGBP', axis=1), one_hot.astype(int)], axis=1)
        one_hot = pd.get_dummies(X_test.IGBP)
        X_test = pd.concat([X_test.drop('IGBP', axis=1), one_hot.astype(int)], axis=1)
        
        model =  RandomForestRegressor(**best_params)
        model.fit(
            X_train, y_train
        )

        y_pred = model.predict(X_test)
        
    elif model == 'CatBoost':
        best_params = {'depth': 16,  'iterations': 2000,
                   'bootstrap_type': 'Poisson', 
                   'l2_leaf_reg': 0, 
                   'loss_function': 'RMSE', 'task_type': 'GPU', 
                   'verbose': 0, 'random_seed': 42}
        
        X_train['IGBP'] = X_train.IGBP.astype('category')
        X_test['IGBP'] = X_test.IGBP.astype('category')

        train_pool = Pool(X_train, y_train, cat_features=['IGBP'])
        test_pool = Pool(X_test, y_test, cat_features=['IGBP'])

        model = CatBoostRegressor(**best_params)
        model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50, verbose=0)
        y_pred = model.predict(X_test)
    elif model == 'ANN':
        num_epoch = 2000
        feature_num = 82
        n1 = 400
        n2 = 100
        dropout=0.3
        BATCH_SIZE = 512
        device = ('cuda' if torch.cuda.is_available() else 'cpu')

        X_train['IGBP'] = X_train.IGBP.astype('category')
        X_test['IGBP'] = X_test.IGBP.astype('category')

        one_hot = pd.get_dummies(X_train.IGBP)
        X_train_ann = pd.concat([X_train.drop('IGBP', axis=1), one_hot.astype(int)], axis=1)
        one_hot = pd.get_dummies(X_test.IGBP)
        X_test_ann = pd.concat([X_test.drop('IGBP', axis=1), one_hot.astype(int)], axis=1)

        train_dataset = Dataset(X_train_ann, y_train.values, fit_scaler=True)
        x_scaler = train_dataset.x_scaler 
        y_scaler = train_dataset.y_scaler 
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=False)
        test_loader = DataLoader(Dataset(X_test_ann.values, y_test.values,
                                         x_scaler=x_scaler, y_scaler=y_scaler), batch_size=BATCH_SIZE, drop_last=False)

        model = ANN(feature_num, n1,n2, dropout).to(device).to(torch.float32)

        criteria = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001) 

        y_pred = train_ann(train_dataset, train_loader, test_loader, model, criteria, optimizer, num_epoch)
    
    return y_pred