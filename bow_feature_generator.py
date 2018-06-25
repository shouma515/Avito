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
import numpy as np

from feature_map import feature_map, feature_map_active

import pickle
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

PICKLE_FOLDER = "pickles/"
TRAIN_TITLE_PATH = PICKLE_FOLDER + 'title'
TEST_TITLE_PATH = PICKLE_FOLDER + 'title_test'
TRAIN_ACTIVE_TITLE_PATH = PICKLE_FOLDER + 'title_active'
TEST_ACTIVE_TITLE_PATH = PICKLE_FOLDER + 'title_active_test'
TRAIN_DESC_PATH = PICKLE_FOLDER + 'description'
TEST_DESC_PATH = PICKLE_FOLDER + 'description_test'
TRAIN_ACTIVE_DESC_PATH = PICKLE_FOLDER + 'description_active'
TEST_ACTIVE_DESC_PATH = PICKLE_FOLDER + 'description_active_test'

TRAIN_SIZE = 1503424
TEST_SIZE = 508438


# Generate the lite version of data, no image, no title, no description.
# Can reduce the image size down to 1/3.
def generate_active_raw():
    train_active_df = pd.read_csv('data/train_active.csv', parse_dates=['activation_date'])
    train_active_df['description'].to_pickle(PICKLE_FOLDER + 'description_active')
    # train_active_df['title'].to_pickle(PICKLE_FOLDER + 'title_active')
    del train_active_df
    print('train active raw generated')

    test_active_df = pd.read_csv('data/test_active.csv', parse_dates=['activation_date'])
    test_active_df['description'].to_pickle(PICKLE_FOLDER + 'description_active_test')
    # test_active_df['title'].to_pickle(PICKLE_FOLDER + 'title_active_test')
    del test_active_df
    print('test active raw generated')

def generate_desc_bow():
    # Reads raw train/test data.
    train = pd.read_pickle(TRAIN_DESC_PATH)
    test = pd.read_pickle(TEST_DESC_PATH)
    # train_active = pd.read_pickle(TRAIN_ACTIVE_DESC_PATH)
    # test_active = pd.read_pickle(TEST_ACTIVE_DESC_PATH)

    _generate_pickle(_bow_desc, train, test, None, None, 'bow_desc')

def generate_title_bow():
    # Reads raw train/test data.
    train = pd.read_pickle(TRAIN_TITLE_PATH)
    test = pd.read_pickle(TEST_TITLE_PATH)
    train_active = pd.read_pickle(TRAIN_ACTIVE_TITLE_PATH)
    test_active = pd.read_pickle(TEST_ACTIVE_TITLE_PATH)

    _generate_pickle(_bow_title, train, test, train_active, test_active, 'bow_title')

def _generate_pickle(func, train, test, train_active, test_active, name):
    # Generates feature series/df for train and test dataset
    train_result, test_result = func(train, test, train_active, test_active)
    # Sanity check
    assert(train_result.shape[0] == TRAIN_SIZE)
    assert(test_result.shape[0] == TEST_SIZE)
    # Renames series so they have proper name when they are used in train/test dataframe.
    with open(PICKLE_FOLDER + name, 'wb') as f:
        pickle.dump(train_result, f)
    with open(PICKLE_FOLDER + name + '_test', 'wb') as f:
        pickle.dump(test_result, f)

    print(name, ' feature generated.')

# BOW features
def _bow_title(train, test, train_active, test_active):
    train_d = train.astype(str).map(_normalize_text)
    test_d = test.astype(str).map(_normalize_text)
    train_active_d = train_active.astype(str).map(_normalize_text)
    test_active_d = test_active.astype(str).map(_normalize_text)
    data_all = pd.concat([train_d, test_d, train_active_d, test_active_d])
    count_vectorizer_title = CountVectorizer(
        stop_words=stopwords.words('russian'), lowercase=True,
        max_df=0.7, ngram_range=(1, 2), max_features=7000)
    count_vectorizer_title.fit(data_all)
    del data_all

    data = pd.concat([train, test])
    title_counts = count_vectorizer_title.fit_transform(data)

    return title_counts[:len(train)], title_counts[len(train):]



def _bow_desc(train, test, train_active, test_active):
    tfidf_para = {
        "analyzer": 'word',
        "token_pattern": r'\w{1,}',
        "sublinear_tf": True,
        # "dtype": np.float32,
        "norm": 'l2',
        #"min_df":5,
        #"max_df":.9,
    #     "smooth_idf":False
    }

    train_d = train.astype(str).map(_normalize_text)
    test_d = test.astype(str).map(_normalize_text)
    data = pd.concat([train_d, test_d])
    count_vectorizer_desc = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), stop_words=stopwords.words('russian'), **tfidf_para,
        max_features=17000)

    desc_counts = count_vectorizer_desc.fit_transform(data)

    return desc_counts[:len(train)], desc_counts[len(train):]

def _normalize_text(text):
    text = text.lower().strip()
    for s in string.punctuation:
        text = text.replace(s, ' ')
    text = text.strip().split(' ')
    return u' '.join(x for x in text if len(x) > 1)


def main():
    # generate_active_raw()
    generate_desc_bow()
    # generate_title_bow()

if __name__ == '__main__':
    main()
