# Train model, generate prediction and submission.
# Sample usage:
#   python train.py -c config_name  # to cross validation
#   python train.py -c config_name -s  # to predict and generate submission:
#       
# TODO: 
# 1. Add option to print erroneous rows in cross validation.

import datetime
import json
import math
import os
import pickle
import time
from optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from configs import config_map
# TODO: move constants used across files to a single file 
from feature_generator import PICKLE_FOLDER, TARGET_PATH
from models import model_map

TRAIN_SIZE = 1503424
TEST_SIZE = 508438
SUBMISSION_FOLDER = 'submissions/'
SUBMISSION_HISTORY_FOLDER = SUBMISSION_FOLDER + 'history/'
RECORD_FOLDER = 'records/'


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
        y = pd.read_pickle(TARGET_PATH)

    # Sanity check
    if test:
        assert(X.shape == (TEST_SIZE, len(feature_names)))
    else:
        assert(X.shape == (TRAIN_SIZE, len(feature_names)))
        assert(y.shape == (TRAIN_SIZE,))
        
    return X, y

# Retrieves the model class from model map and creates an instance of it.
def get_model(model_name, model_params):
    return model_map[model_name](model_params=model_params)

# Returns two array containing validation and train errors of each fold.
def cross_validate(config, X, y):
    model_name = config['model']
    model_params = config['model_params']
    folds = config['folds']

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    train_errors = []
    val_errors = []
    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        print('Fold ', i)

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val,y_val = X.iloc[val_index], y.iloc[val_index]

        print('training...')
        model = get_model(model_name, model_params)
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

# Separates the cross validation with the data preparation step. The main purpose is
# we do not need to repeat data preparation when tuning a model.
def train(config, record=True):
    # Prepare train data.    
    feature_names = config['features']
    X, y = prepare_data(feature_names, test=False)
    # For debug use only
    # print(X.columns)
    # print(X.describe(include='all'))
    # print(y.describe())
    # print(X.head())
    # print(y.head())
    # print(X.index)
    # print(y.index)
    # X = X[:500]
    # y = y[:500]
    val_erros, train_errors = cross_validate(config, X, y)
    # Records the cross validation in a json file if needed.
    # TODO: figure out query json file for analysis.
    if record:
        # Remove tune_params from config, as it is not serializable, and we do
        # not need to record it.
        config.pop('tune_params')
        print(config)
        record_dict = {
            'config': config,
            'train_errors': train_errors,
            'val_errors': val_erros
        }
        timestamp = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")    
        if not os.path.exists(RECORD_FOLDER):
            os.makedirs(RECORD_FOLDER)
        with open(
            '%s%s_%s' %(RECORD_FOLDER, config['name'], timestamp),
            'w'
        ) as fp:
            json.dump(record_dict, fp)


def predict(config):
    feature_names = config['features']
    model_name = config['model']
    model_params = config['model_params']
    
    # Prepares train data.
    X_train, y_train = prepare_data(feature_names, test=False)
    X_test, _ = prepare_data(feature_names, test=True)

    print('training...')
    model = get_model(model_name, model_params)
    model.fit(X_train, y_train)
    rmse = math.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    print('training error: ', rmse)

    print('predicting...')
    prediction = model.predict(X_test)
    # Clips predictions to be between 0 and 1.
    np.clip(prediction, 0, 1, out=prediction)
    # Sanity check.
    assert(len(prediction) == TEST_SIZE)
    
    submission = pd.read_csv('data/sample_submission.csv')
    # Sample submission file and test dataset has the same item_id
    # in the same order.
    submission['deal_probability'] = prediction

    # Timestamp for naming of the submission files.
    sub_timestamp = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
    # Submission history folder is a sub directory of submission folder, thus
    # the following command will create both on the way.
    if not os.path.exists(SUBMISSION_HISTORY_FOLDER):
        os.makedirs(SUBMISSION_HISTORY_FOLDER)
    # Generates submission csv.
    submission.to_csv(
        '%s%s_%s.csv' %(SUBMISSION_FOLDER, config['name'], sub_timestamp),
        index=False
    )
    # Saves the submission and its config as pickle for future investigation.
    submission_history = {
        'config': config,
        'submission': submission
    }
    sub_history_file = open(
        '%s%s_%s' %(
            SUBMISSION_HISTORY_FOLDER, config['name'], sub_timestamp),
        'wb')
    pickle.dump(submission_history, sub_history_file)

if __name__ == '__main__':
    t_start = time.time()
    # parser to parse cmd line option
    parser = OptionParser()
    # add options to parser, currently only config file
    parser.add_option(
        '-c', '--config', action='store', type='string', dest='config')
    parser.add_option(
        '-s', '--submit', action='store_true', dest='submit', default=False)
    
    options, _ = parser.parse_args()
    config = config_map[options.config]
    # Adds a name fields for the naming of submission / record files.
    config['name'] = options.config
    if options.submit:
        # Predicts on test set and generates submission files.
        predict(config)
    else:
        # Cross validation.
        train(config)

    t_finish = time.time()
    print('Total running time: ', (t_finish - t_start) / 60)
