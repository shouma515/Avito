#TODO:
# 1. encode categorical data, especially image_top_1

import argparse
import os

import pandas as pd

from feature_map import feature_map

PICKLE_FOLDER = "pickles/"
TRAIN_PICKLE_PATH = PICKLE_FOLDER + 'df_train'
TEST_PICKLE_PATH = PICKLE_FOLDER + 'df_test'


def generate_raw_df_pickle():
    # train dataset
    train_df = pd.read_csv('data/train.csv', parse_dates=['activation_date'])
    train_df.to_pickle(TRAIN_PICKLE_PATH)
    # test dataset
    test_df = pd.read_csv('data/test.csv', parse_dates=['activation_date'])
    test_df.to_pickle(TEST_PICKLE_PATH)

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
        help='If just generates pickle for train or test (decided by --test flag) dataset.')    
    args = parser.parse_args()

    if not os.path.exists(PICKLE_FOLDER):
        os.makedirs(PICKLE_FOLDER)

    if args.raw:
        generate_raw_df_pickle()
    else:
        feature_names = args.feature_names
        if args.all:
            feature_names = list(feature_map.keys())
        generate_features(feature_names)


if __name__ == '__main__':
    main()
