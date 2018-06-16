# Generates pickles of train/test datasets and features.
# Train/target/test and feature pickles need to be generated first before used
# in training or predicting.
#
# Sample usage:
#   python feature_generator.py --raw  # to generate pickles of train and test
#                                        datasets and train traget
#   python feature_generator.py --active  # to generate pickles of train active
#                                        and test active datasets
#   python feature_generator.py --all  # to generate pickles of all the features
#                                        in the feature map
#   python feature_generator.py --all --active # to generate pickles of all the
#                                           features in the active feature map,
#                                           i.e. including active datasets
#   python feature_generator.py feature_1, feature_2,...
#   # to generate pickles for features listed, For one call, the, features need
#   # to be all in active feature map or normal feature map. Can not be mixed.


import argparse
import os

import pandas as pd

from feature_map import feature_map, feature_map_active

PICKLE_FOLDER = "pickles/"
TRAIN_PICKLE_PATH = PICKLE_FOLDER + 'df_train'
TEST_PICKLE_PATH = PICKLE_FOLDER + 'df_test'
TRAIN_LITE_PICKLE_PATH = PICKLE_FOLDER + 'df_train_lite'
TEST_LITE_PICKLE_PATH = PICKLE_FOLDER + 'df_test_lite'
TRAIN_ACTIVE_LITE_PICKLE_PATH = PICKLE_FOLDER + 'df_train_active_lite'
TEST_ACTIVE_LITE_PICKLE_PATH = PICKLE_FOLDER + 'df_test_active_lite'
TARGET_COLUMN = 'deal_probability'
TARGET_PATH = PICKLE_FOLDER + TARGET_COLUMN

TRAIN_SIZE = 1503424
TEST_SIZE = 508438


# Generate pickle for train/test dataset, train target.
def generate_raw_df_pickle():
    # train dataset
    train_df = pd.read_csv('data/train.csv', parse_dates=['activation_date'])
    train_df.to_pickle(TRAIN_PICKLE_PATH)
    train_df[TARGET_COLUMN].to_pickle(TARGET_PATH)
    # test dataset
    test_df = pd.read_csv('data/test.csv', parse_dates=['activation_date'])
    test_df.to_pickle(TEST_PICKLE_PATH)

# Generate pickles for train, test, train_active and test_active data,
# lite version.
def generate_lite_pickle():
    # train_active dataset lite
    train_active_df = pd.read_csv('data/train_active_lite.csv', parse_dates=['activation_date'])
    train_active_df.to_pickle(TRAIN_ACTIVE_LITE_PICKLE_PATH)
    del train_active_df
    # test_active dataset lite
    test_active_df = pd.read_csv('data/test_active_lite.csv', parse_dates=['activation_date'])
    test_active_df.to_pickle(TEST_ACTIVE_LITE_PICKLE_PATH)
    del test_active_df
    # train  lite
    train_lite_df = pd.read_csv('data/train_lite.csv', parse_dates=['activation_date'])
    train_lite_df.to_pickle(TRAIN_LITE_PICKLE_PATH)
    del train_lite_df
    # test dataset lite
    test_lite_df = pd.read_csv('data/test_lite.csv', parse_dates=['activation_date'])
    test_lite_df.to_pickle(TEST_LITE_PICKLE_PATH)
    del test_lite_df

# Generate the lite version of data, no image, no title, no description.
# Can reduce the image size down to 1/3.
def generate_lite():
    lite_columns = ['item_id', 'user_id', 'region', 'city',
        'parent_category_name', 'category_name', 'param_1', 'param_2',
        'param_3', 'price', 'item_seq_number', 'activation_date', 'user_type']

    train_df = pd.read_csv('data/train.csv', parse_dates=['activation_date'])
    train_df[lite_columns + [TARGET_COLUMN]].to_csv(
        'data/train_lite.csv', index=False)
    del train_df

    test_df = pd.read_csv('data/test.csv', parse_dates=['activation_date'])
    test_df[lite_columns].to_csv('data/test_lite.csv', index=False)
    del test_df

    train_active_df = pd.read_csv('data/train_active.csv', parse_dates=['activation_date'])
    train_active_df[lite_columns].to_csv('data/train_active_lite.csv', index=False)
    del train_active_df

    test_active_df = pd.read_csv('data/test_active.csv', parse_dates=['activation_date'])
    test_active_df[lite_columns].to_csv('data/test_active_lite.csv', index=False)
    del test_active_df


def generate_features(name_list):
    # Reads raw train/test data.
    train_df = pd.read_pickle(TRAIN_PICKLE_PATH)
    test_df = pd.read_pickle(TEST_PICKLE_PATH)

    for name in name_list:
        try:
            generate_feature_pickle(name, train_df, test_df)
        except KeyError as e:
            # If a feature name is not in the feature map.
            print(e + 'not found in feature map.')

def generate_feature_pickle(name, train_df, test_df):
    # Generates feature series/df for train and test dataset
    train, test = feature_map[name](train_df, test_df)
    # Renames series so they have proper name when they are used in train/test dataframe.
    if isinstance(train, pd.Series):
        train.rename(name, inplace=True)
        test.rename(name, inplace=True)
    # Sanity check
    assert(train.shape[0] == TRAIN_SIZE)
    assert(test.shape[0] == TEST_SIZE)

    # Generates pickle.
    pickle_path = PICKLE_FOLDER + name
    train.to_pickle(pickle_path)
    test.to_pickle(pickle_path + '_test')

    print(name, ' feature generated.')

def generate_active_features(name_list):
    # Use lite dataset when using active data.
    train_lite_df = pd.read_pickle(TRAIN_LITE_PICKLE_PATH)
    test_lite_df = pd.read_pickle(TEST_LITE_PICKLE_PATH)
    train_active_lite_df = pd.read_pickle(TRAIN_ACTIVE_LITE_PICKLE_PATH)
    test_active_lite_df = pd.read_pickle(TEST_ACTIVE_LITE_PICKLE_PATH)

    for name in name_list:
        try:
            generate_active_feature_pickle(name, train_lite_df, test_lite_df, train_active_lite_df, test_active_lite_df)
        except KeyError as e:
            # If a feature name is not in the feature map.
            print(e + 'not found in feature map.')

def generate_active_feature_pickle(name, train_df, test_df, train_active_df, test_active_df):
    # Generates feature series/df for train and test dataset
    train, test = feature_map_active[name](train_df, test_df, train_active_df, test_active_df)
    # Renames series so they have proper name when they are used in train/test dataframe.
    if isinstance(train, pd.Series):
        train.rename(name, inplace=True)
        test.rename(name, inplace=True)

    # Generates pickle.
    pickle_path = PICKLE_FOLDER + name
    train.to_pickle(pickle_path)
    test.to_pickle(pickle_path + '_test')

    print(name, ' feature generated.')


def main():
    parser = argparse.ArgumentParser(description='Process the features to generate.')
    parser.add_argument('feature_names', metavar='F', type=str, nargs='*',
        help='Name of a feature to be generated.')
    parser.add_argument('--all', dest='all', action='store_true', default=False,
        help='If generates all features in the feature map.')
    parser.add_argument('--raw', dest='raw', action='store_true', default=False,
        help='If just generates pickle for train and test dataset.')
    parser.add_argument('--active', dest='active', action='store_true', default=False,
        help='If just generates pickles for/using train and test active dataset.')
    args = parser.parse_args()

    if not os.path.exists(PICKLE_FOLDER):
        os.makedirs(PICKLE_FOLDER)

    if args.raw:
        if args.active:
            generate_lite()
            generate_lite_pickle()
        else:
            generate_raw_df_pickle()

    else:
        feature_names = args.feature_names
        if args.active:
            if args.all:
                feature_names = list(feature_map_active.keys())
            generate_active_features(feature_names)
        else:
            if args.all:
                feature_names = list(feature_map.keys())
            generate_features(feature_names)


if __name__ == '__main__':
    main()
