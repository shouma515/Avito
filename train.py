# TODO: 
# 1. generate submission
# 2. Add record

import math
import time
from optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from configs import config_map
# TODO: move constants used across files to a single file 
from feature_generator import PICKLE_FOLDER
from models import model_map

TRAIN_SIZE = 1503424

# Prepares train/test data. Target column will be returned when get train data;
# when get test data, it will be None.
# Two things to note:
#   1. All the feature pickles should be generated before training;
#   2. Order of training data should not be changed if you want to
#      compare result between trainings, as cross validation depends
#      on that.
def prepare_data(feature_names, test=False):
    features = []
    for name in feature_names:
        # Assume all the feature pickles are generated. Any features not
        # generated will cause an error here.
        pickle_path = PICKLE_FOLDER + name
        if test:
            pickle_path += '_test'
        features.append(pd.read_pickle(pickle_path))
    
    X = pd.concat(features, axis=1)
    y = None
    if not test:
        y = pd.read_pickle(PICKLE_FOLDER + 'deal_probability')

    # Sanity check
    assert(X.shape == (TRAIN_SIZE, len(feature_names)))
    if not test:
        assert(y.shape == (TRAIN_SIZE,))
        
    return X, y


def get_model(model_name, model_params):
    return model_map[model_name](model_params=model_params)


def train(config):
    feature_names = config['features']
    model_name = config['model']
    model_params = config['model_params']
    folds = config['folds']
    
    # Prepare train data.
    X, y = prepare_data(feature_names, test=False)
    
    # For debug use only
    # print(X.describe(include='all'))
    # print(y.describe())
    # print(X.head())
    # print(y.head())
    # print(X.index)
    # print(y.index)
    X = X[:500]
    y = y[:500]

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    val_error = 0
    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        print('Fold ', i)

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val,y_val = X.iloc[val_index], y.iloc[val_index]

        print('training...')
        model = get_model(model_name, model_params)
        model.fit(X_train, y_train)

        print('validating...')
        rmse = math.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        print('validation error: ', rmse)
        val_error += rmse
        
        print('-----------------------------------------')
    
    print('\nAvg validation error: ', val_error / folds)


if __name__ == '__main__':
    t_start = time.time()
    # parser to parse cmd line option
    parser = OptionParser()
    # add options to parser, currently only config file
    parser.add_option('-c', '--config', action='store', type='string', dest='config')
    parser.add_option('-s', '--submit', action='store_true', dest='submit', default=False)
    
    options, _ = parser.parse_args()
    config = config_map[options.config]
    train(config)

    t_finish = time.time()
    print('Total running time: %f min', (t_finish - t_start) / 60)
