from train import prepare_data, get_model
from sklearn.model_selection import KFold
import os
import time
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from configs import config_map

def ensemble_for_lgb(config, X, y):
    model_name = config['model']
    model_params = config['model_params']

    # Cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    predicts = []
    val_errors = []
    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        print('Fold ', i)

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val,y_val = X.iloc[val_index], y.iloc[val_index]

        X_train = X_train.drop('item_id', axis=1)
        val_id = X_val['item_id']
        X_val = X_val.drop('item_id', axis=1)
        # debug info
        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)

        print('training...')
        model = get_model(model_name, model_params.copy(), config)
        model.fit(X_train, y_train)
        predict = model.predict(X_val)
        np.clip(predict, 0, 1, out=predict)

        val_error = math.sqrt(mean_squared_error(y_val, predict))
        print('validate caculated: %f' %val_error)
        val_errors.append(val_error)

        predict = pd.Series(predict, name='predict')
        val_id = val_id.reset_index(drop=True)
        predict = pd.concat([val_id, predict], axis=1)
        predicts.append(predict)
    return predicts, val_errors

config = config_map['catboost_config']
ENSEMBLE_FOLDER = 'ensemble/'

X, y = prepare_data(config['features'] + ['item_id'], config['image_feature_folders'], test=False)
predicts, val_errors = ensemble_for_lgb(config, X, y)

print('Avg validation error: %f' %(np.mean(val_errors)))
for predict in predicts:
    print(predict.shape)
ensembled = pd.concat(predicts)
print(ensembled.shape)
assert(ensembled.shape == (len(X), 2))

if not os.path.exists(ENSEMBLE_FOLDER):
        os.makedirs(ENSEMBLE_FOLDER)
ensembled.to_csv(ENSEMBLE_FOLDER+'catboost.csv', index=False)