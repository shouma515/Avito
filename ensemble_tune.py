# Tuning a model
# Sample usage:
#   python tune.py -c config_name  # to tune parameters for the given configuration
#   python tune.py -c config_name -t trials_file  # to continue tune parameters for
#                                                   the given config, starting from
#                                                   the given trials.

# If encounter "TypeError: 'generator' object is not subscriptable",
# try pip install --upgrade git+git://github.com/hyperopt/hyperopt.git
# see https://github.com/hyperopt/hyperopt/pull/319
import os
import pickle
import time
from datetime import datetime
from optparse import OptionParser

import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe

from configs import config_map
from feature_generator import TARGET_PATH

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math

TRIALS_FOLDER = 'ensemble/trials/'
ENSEMBLE_FOLDER = 'ensemble/'

def cross_validate(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_errors = []
    val_errors = []
    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        print('Fold ', i)

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val,y_val = X.iloc[val_index], y.iloc[val_index]

        print('training...')
        model = LinearRegression()
        model.fit(X_train, y_train)
        train_rmse = math.sqrt(
            mean_squared_error(y_train, model.predict(X_train)))
        print('training error: ', train_rmse)
        train_errors.append(train_rmse)

        print('validating...')
        rmse = math.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        print('validation error: ', rmse)
        val_errors.append(rmse)

        print('-----------------------------------------')

    print('\nAvg validation error: ', np.mean(val_errors))
    return val_errors, train_errors

def tune_single_model(parameter_space, config_name, max_evals, trials=None):
    # Prepare train data.
    ensemble_csv = ['catboost.csv', 'lgb.csv', 'xgboost.csv']
    X = pd.read_csv(ENSEMBLE_FOLDER + ensemble_csv[0])
    for i in range(1, len(ensemble_csv)):
        pred = pd.read_csv(ENSEMBLE_FOLDER + ensemble_csv[1])
        X = X.merge(pred, 'left', on='item_id', suffixes=('', str(i)))

    y = pd.read_pickle(TARGET_PATH)[:10000]

    def train_wrapper(params):
        cv_losses, cv_train_losses = cross_validate(X, y)
        # return an object to be recorded in hyperopt trials for future uses
        return {
            'loss': np.mean(cv_losses),
            'losses': cv_losses,
            'train_loss': cv_train_losses,
            'status': STATUS_OK,
            'eval_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'params': params
        }

    if trials is None:
        trials = Trials()
    # tuning parameters
    t1 = time.time()
    timestamp = datetime.now().strftime("%m-%d_%H:%M:%S")
    best = fmin(train_wrapper, parameter_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    t2 = time.time()
    print('best trial get at round: ' + str(trials.best_trial['tid']))
    print('best loss: ' + str(trials.best_trial['result']['loss']))
    print(best)
    print(space_eval(parameter_space, best))
    print("time: %s" %((t2-t1) / 60))

    # save the experiment trials in a pickle
    if not os.path.exists(TRIALS_FOLDER):
        os.makedirs(TRIALS_FOLDER)
    # TODO: save tuning config when dump trials pickle.
    pickle.dump(trials, open("%s%s_%s" %(TRIALS_FOLDER, config_name, timestamp), "wb"))

    return trials

def main():
    ensemble_csv = ['catboost.csv', 'lgb.csv', 'xgboost.csv']
    X = pd.read_csv(ENSEMBLE_FOLDER + ensemble_csv[0])
    for i in range(1, len(ensemble_csv)):
        pred = pd.read_csv(ENSEMBLE_FOLDER + ensemble_csv[1])
        X = X.merge(pred, 'left', on='item_id', suffixes=('', str(i)))

    print(X.columns)
    X.drop('item_id', axis=1, inplace=True)

    y = pd.read_pickle(TARGET_PATH)
    item_id = pd.read_pickle(item_id_pickle_path).to_frame()

    print(X.shape)
    print(y.shape)
    cv_losses, cv_train_losses = cross_validate(X, y)
    print(cv_losses)
    print(cv_train_losses)
    print(np.mean(cv_losses))

if __name__ == '__main__':
    main()
