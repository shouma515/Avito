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
import random
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
RECORD_FOLDER = 'records/'
CV_RECORD_FOLDER = RECORD_FOLDER + 'cv/'
SUBMISSION_RECORD_FOLDER = RECORD_FOLDER + 'sub/'
MODEL_PICKLE_FOLDER = RECORD_FOLDER + 'model/'


# Prepares train/test data. Target column will be returned when get train data;
# when get test data, it will be None.
# Two things to note:
#   1. All the feature pickles should be generated before training;
#   2. Order of training data should not be changed if you want to
#      compare result between trainings, as cross validation depends
#      on that.
def prepare_data(feature_names, image_feature_folders=[], test=False):
    DATA_LENTH = TEST_SIZE if test else TRAIN_SIZE

    features = []
    total_feature = 0
    for name in feature_names:
        # Assume all the feature pickles are generated. Any features not
        # generated will cause an error here.
        pickle_path = PICKLE_FOLDER + name
        if test:
            pickle_path += '_test'
        feature = pd.read_pickle(pickle_path)
        # Sanity check
        assert(feature.shape[0] == DATA_LENTH)
        if isinstance(feature, pd.DataFrame):
            total_feature += feature.shape[1]
        else:
            # Series
            total_feature += 1

        features.append(feature)

    # Add image features.
    if len(image_feature_folders) > 0:
        # load item id to join image features.
        item_id_pickle_path = PICKLE_FOLDER + 'item_id'
        if test:
            item_id_pickle_path += '_test'
        # Need to convert the item_id Series to DataFrame for merge with image
        # features.
        item_id = pd.read_pickle(item_id_pickle_path).to_frame()
        # Sanity check
        assert(item_id.shape[0] == DATA_LENTH)
        image_features = load_image_features(image_feature_folders, test)
        image_features = item_id.merge(
            image_features, how='left', on='item_id', validate='1:1')
        image_features.drop('item_id', axis=1, inplace=True)
        # Sanity check
        assert(image_features.shape[0] == DATA_LENTH)
        total_feature += image_features.shape[1]
        features.append(image_features)

    X = pd.concat(features, axis=1)
    y = None
    if not test:
        y = pd.read_pickle(TARGET_PATH)


    # Sanity check
    assert(X.shape == (DATA_LENTH, total_feature))
    print("Data size:", X.shape)
    if not test:
        assert(y.shape == (TRAIN_SIZE,))
        print("Label size:", y.shape)

    # Debug info
    print(X.columns)
    print(X.shape)
    if not test:
        print(y.name)
        print(y.shape)

    X = reduce_mem_usage(X)

    return X, y


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    simple = 0
    int_long = 0
    float_long = 0
    object_type = 0

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            if col_type == 'bool' or col_type == 'int8' or col_type == 'int16':
                simple += 1
                continue
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                int_long += 1
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                assert('float' in str(col_type))
                float_long += 1
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            object_type += 1
            print(col, ':', col_type)

    print(simple, int_long, float_long, object_type, simple+int_long+float_long+object_type, len(df.columns))
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# Each set of image features is in one folder. And we load features folder by
# folder and join them with item_id.
def load_image_features(image_feature_folders, test):
    assert(len(image_feature_folders) > 0)
    folder0 = image_feature_folders[0]
    image_features = load_image_feature(folder0, test)
    for folder in image_feature_folders[1:]:
        feature = load_image_feature(folder, test)
        image_features.merge(feature, how='left', on='item_id', validate='1:1')

    return image_features

# Each image feature folder should contain a schema file(with names of the
# columns in the feature files) and a pair of
# image_feature.csv, image_feature_test.csv file.
# The features need to have item_id and image_id as primay key.
def load_image_feature(folder, test):
    # # debug
    # print('image feature', folder)

    with open(folder + '/schema', 'r') as schema_in:
            schema = schema_in.read().strip()
    schema = schema.split(',')

    image_feature_path = folder + '/image_feature.csv'
    if test:
        image_feature_path = folder + '/image_feature_test.csv'
    image_feature = pd.read_csv(image_feature_path, header=None, names=schema)

    # image ids is used debug, we don't use them in training.
    image_feature.drop('image', axis=1, inplace=True)
    return image_feature


# Retrieves the model class from model map and creates an instance of it.
def get_model(model_name, model_params):
    return model_map[model_name](model_params=model_params)


# TODO: figure out query json file for analysis.
# TODO: put utility functions in a separate file.
# Note that record_cv will change config (remove tune params), but it shouldn't
# matter in training and predicting.
# TODO: figure out a better way to handle tuning parameters.
def record_cv(
    config, val_errors, train_errors,
    timestamp=datetime.datetime.now().strftime("%m-%d_%H:%M:%S")):
    # Remove tune_params from config, as it is not serializable, and we do
    # not need to record it.
    if 'tune_params' in config:
        config.pop('tune_params')
    record_dict = {
        'config': config,
        'train_errors': train_errors,
        'val_errors': val_errors,
        'sub_error': 0 # Need to fill manually after submission.
    }
    if not os.path.exists(CV_RECORD_FOLDER):
        os.makedirs(CV_RECORD_FOLDER)
    with open(
        '%s%s_%s' %(CV_RECORD_FOLDER, config['name'], timestamp),
        'w'
    ) as fp:
        json.dump(record_dict, fp)


# Returns two array containing validation and train errors of each fold.
def cross_validate(config, X, y):
    return cross_validate_strategy_1(config, X, y, 0.25)

def cross_validate_strategy_0(config, X, y):
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

# Use 25% data to form cv set. Thus, if 5 folds, in every fold, 25/5 = 5%
# will be used as validate, other 95% will be used for train.
def cross_validate_strategy_1(config, X, y, cv_percent):
    model_name = config['model']
    model_params = config['model_params']
    folds = config['folds']

    # Split a part of data to form cv set, other data will be used in all
    # trains.
    total = len(X)
    idx = list(range(total))
    random.seed(42)
    random.shuffle(idx)

    cv_set_size = int(total * cv_percent)
    cv_set_idx = idx[:cv_set_size]
    X_cv_set, y_cv_set = X.iloc[cv_set_idx], y.iloc[cv_set_idx]

    other_idx = idx[cv_set_size:]
    X_other, y_other = X.iloc[other_idx], y.iloc[other_idx]

    # Cross validation
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    train_errors = []
    val_errors = []
    for i, (train_index, val_index) in enumerate(kf.split(X_cv_set, y_cv_set)):
        print('Fold ', i)

        X_train, y_train = X_cv_set.iloc[train_index], y_cv_set.iloc[train_index]
        X_train = pd.concat([X_train, X_other])
        y_train = pd.concat([y_train, y_other])
        X_val, y_val = X_cv_set.iloc[val_index], y_cv_set.iloc[val_index]
        # debug info
        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)

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

    print('\nAvg train error: ', np.mean(train_errors))
    print('Avg validation error: ', np.mean(val_errors))
    print('----------------------------------------\n')
    return val_errors, train_errors


# Separates the cross validation with the data preparation step. The main purpose is
# we do not need to repeat data preparation when tuning a model.
def train(config, record=True):
    # Prepare train data.
    feature_names = config['features']
    image_feature_folders = config['image_feature_folders']
    X, y = prepare_data(feature_names, image_feature_folders, test=False)
    return
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
    val_errors, train_errors = cross_validate(config, X, y)
    # Records the cross validation in a json file if needed.
    if record:
        record_cv(config, val_errors, train_errors)


# Predicts on test data and generates submission.
def predict(config, cv=True):
    feature_names = config['features']
    image_feature_folders = config['image_feature_folders']
    model_name = config['model']
    model_params = config['model_params']

    # Prepares train data.
    X_train, y_train = prepare_data(
        feature_names, image_feature_folders, test=False)
    X_test, _ = prepare_data(feature_names, image_feature_folders, test=True)

    # Timestamp for naming of the submission file and the cv record file.
    sub_timestamp = datetime.datetime.now().strftime("%m-%d_%H:%M:%S")

    # Cross-validates with the config to record the local validation errors.
    if cv:
        print('Cross validating and recording local cv result...')
        val_errors, train_errors = cross_validate(config, X_train, y_train)
        record_cv(config, val_errors, train_errors, sub_timestamp)

    print('training on entire dataset...')
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

    # Submission history folder is a sub directory of submission folder, thus
    # the following command will create both on the way.
    if not os.path.exists(SUBMISSION_RECORD_FOLDER):
        os.makedirs(SUBMISSION_RECORD_FOLDER)

    if not os.path.exists(SUBMISSION_FOLDER):
        os.makedirs(SUBMISSION_FOLDER)

    if not os.path.exists(MODEL_PICKLE_FOLDER):
        os.makedirs(MODEL_PICKLE_FOLDER)

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
            SUBMISSION_RECORD_FOLDER, config['name'], sub_timestamp),
        'wb')
    pickle.dump(submission_history, sub_history_file)

    model_file = open(
        '%s%s_%s' %(
            MODEL_PICKLE_FOLDER, config['name'], sub_timestamp),
        'wb')
    pickle.dump(model, model_file)
    # TODO: use kaggle cmd line api to submit and get result.
    # TODO: centralize records, now we have cv records (in json) and
    #       submission history (in pickle).


if __name__ == '__main__':
    t_start = time.time()
    # Parser to parse cmd line option
    parser = OptionParser()
    # Adds options to parser, currently only config file
    parser.add_option(
        '-c', '--config', action='store', type='string', dest='config')
    parser.add_option(
        '-s', '--submit', action='store_true', dest='submit', default=False)
    # Skips cross validation and record when generate submission.
    parser.add_option(
        '-n', '--no_cv', action='store_false', dest='cv', default=True)

    options, _ = parser.parse_args()
    config = config_map[options.config]
    # Adds a name fields for the naming of submission / record files.
    config['name'] = options.config
    if options.submit:
        # Predicts on test set and generates submission files.
        predict(config, options.cv)
    else:
        # Cross validation.
        train(config)

    t_finish = time.time()
    print('Total running time: ', (t_finish - t_start) / 60)
