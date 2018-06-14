# All the functions in this file assume takes the raw train and test dataframes,
# and output a pair of serieses or dataframes contain one or more features
# generated for train and test datasets.
# IMPORTANT: all the functions here should be pure functions, i.e. they should
#            not modify the input train or test dataframes.

import numpy as np
import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords


# Global variables
stopwords = {x: 1 for x in stopwords.words('russian')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')
non_alphanumpunct = re.compile(u'[^A-Za-z0-9\.?!,; \(\)\[\]\'\"\$]+')
RE_PUNCTUATION = re.compile('|'.join([re.escape(x) for x in string.punctuation]))
RE_NUMBER = re.compile('\d+')


# Original fields

# Primary key.
def item_id(train, test):
    return train['item_id'], test['item_id']

def user_id(train, test):
    return _encode_column(train, test, 'user_id')

def region(train, test):
    return _encode_column(train, test, 'region')

def city(train, test):
    return _encode_column(train, test, 'city')

def parent_category_name(train, test):
    return _encode_column(train, test, 'parent_category_name')

def category_name(train, test):
    return _encode_column(train, test, 'category_name')

def param_1(train, test):
    return _encode_column(train, test, 'param_1')

def param_2(train, test):
    return _encode_column(train, test, 'param_2')

def param_3(train, test):
    return _encode_column(train, test, 'param_3')

def title(train, test):
    return train['title'], test['title']

def description(train, test):
    return train['description'], test['description']

def price(train, test):
    return train['price'], test['price']

def item_seq_number(train, test):
    return train['item_seq_number'], test['item_seq_number']

# TODO: some model does not take date time, consider how to deal with them.
def activation_date(train, test):
    return train['activation_date'], test['activation_date']

def user_type(train, test):
    return _encode_column(train, test, 'user_type')

def image(train, test):
    return train['image'], test['image']

def image_top_1(train, test):
    return _encode_column(train, test, 'image_top_1')



# Generated fields

# Counts
# Number of listings per column entry. Calculated with both train and test data.
# Number of listings per user_id.
def listings_per_user(train, test):
    return _counts(train, test, 'user_id')

def listings_per_city_date(train, test):
    return _multi_counts(train, test, ['city', 'activation_date'])

# Aggregation over some dimensions
def price_city_date_mean_max(train, test):
    return _aggregate(
        train, test, ['city', 'activation_date'], ['price'], ['mean', 'max'])

def city_date_price_mean_max_active(train, test, train_active, test_active):
    return _aggregate_active(
        train, test, train_active, test_active, ['city', 'activation_date'], ['price'], ['mean', 'max'])

# Boolean features
# Actually this can be implied by image_top_1 != nan
def has_image(train, test):
    return train['image'].notnull(), test['image'].notnull()

# Text meta features
# Length features
# Title length
def title_len(train, test):
    return train['title'].map(len), test['title'].map(len)

# Title word counts
def title_wc(train, test):
    return train['title'].map(str.split).map(len), test['title'].map(str.split).map(len)

# Number of exclamation point in title
def title_exclam_count(train, test):
    def exclam_count(line):
        return line.count('!')
    return train['title'].map(exclam_count), test['title'].map(exclam_count)

# Number of upper case letters
def title_upper_count(train, test):
    def upper_count(line):
        return len([x for x in line if x.isupper()])
    return train['title'].map(upper_count), test['title'].map(upper_count)

# Ratio of upper case letters in title
def title_upper_count_ratio(train, test):
    def upper_count_ratio(line):
        if len(line) == 0:
            return 0
        return len([x for x in line if x.isupper()]) / len(line)
    return train['title'].map(upper_count_ratio), test['title'].map(upper_count_ratio)


# Number of numbers in title
def title_num_count(train, test):
    def num_count(line):
        return len(RE_NUMBER.findall(line))
    return train['title'].map(num_count), test['title'].map(num_count)

# Ratio of numbers in title
def title_num_count_ratio(train, test):
    def num_count_ratio(line):
        if len(line) == 0:
            return 0
        return len(RE_NUMBER.findall(line)) / len(line)
    return train['title'].map(num_count_ratio), test['title'].map(num_count_ratio)

# Number of unique words in title
def title_uniq_wc(train, test):
    def uniq_wc(line):
        word_list = _normalize_text_word_list(line)
        return len(set(word_list))
    return train['title'].map(uniq_wc), test['title'].map(uniq_wc)

# Ratio of unique words in title
def title_uniq_wc_ratio(train, test):
    def uniq_wc_ratio(line):
        word_list = _normalize_text_word_list(line)
        if len(word_list) == 0:
            return 0
        return len(set(word_list)) / len(word_list)
    return train['title'].map(uniq_wc_ratio), test['title'].map(uniq_wc_ratio)

# Description length
def desc_len(train, test):
    def norm_len(line):
        line = _normalize_text(line)
        return len(line)
    train_desc_len = train['description'].map(str).map(norm_len)
    # Set missing description length to 0. Otherwise they will be 3 ('nan').
    train_desc_len[train['description'].isnull()] = 0

    test_desc_len = test['description'].map(str).map(norm_len)
    test_desc_len[test['description'].isnull()] = 0

    return train_desc_len, test_desc_len

# Description counts
def desc_wc(train, test):
    train_desc_wc = train['description'].map(str).map(str.split).map(len)
    # Set missing description length to 0. Otherwise they will be 1 (['nan']).
    train_desc_wc[train['description'].isnull()] = 0

    test_desc_wc = test['description'].map(str).map(str.split).map(len)
    test_desc_wc[test['description'].isnull()] = 0

    return train_desc_wc, test_desc_wc

# Normalized description counts
def desc_wc_norm(train, test):
    def norm_wc(line):
        word_list = _normalize_text_word_list(line)
        return len(word_list)
    train_desc_wc = train['description'].map(str).map(norm_wc)
    # Set missing description length to 0. Otherwise they will be 1 (['nan']).
    train_desc_wc[train['description'].isnull()] = 0

    test_desc_wc = test['description'].map(str).map(norm_wc)
    test_desc_wc[test['description'].isnull()] = 0

    return train_desc_wc, test_desc_wc

# Normalized wc to all wc ratio
def desc_wc_norm_ratio(train, test):
    def norm_ratio(line):
        norm_word_list = _normalize_text_word_list(line)
        word_list = line.split()
        if len(word_list) == 0:
            return 0
        return len(norm_word_list) / len(word_list)
    train_desc_wc_ratio = train['description'].map(str).map(norm_ratio)
    # Set missing description length to 0. Otherwise they will be 1 (['nan']).
    train_desc_wc_ratio[train['description'].isnull()] = 0

    test_desc_wc_ratio = test['description'].map(str).map(norm_ratio)
    test_desc_wc_ratio[test['description'].isnull()] = 0

# Number of exclamation point in description
def desc_exclam_count(train, test):
    def exclam_count(line):
        return line.count('!')
    return (train['description'].map(str).map(exclam_count),
            test['description'].map(str).map(exclam_count))

# Number of punctuations in description
def desc_punc_count(train, test):
    def punc_count(line):
        return len(RE_PUNCTUATION.findall(line))
    return (train['description'].map(str).map(punc_count),
            test['description'].map(str).map(punc_count))

# Number of upper case letters in description
def desc_upper_count(train, test):
    def upper_count(line):
        return len([x for x in line if x.isupper()])
    return (train['description'].map(str).map(upper_count),
            test['description'].map(str).map(upper_count))

# Ratio of upper case letters in description
def desc_upper_count_ratio(train, test):
    def upper_count_ratio(line):
        if len(line) == 0:
            return 0
        return len([x for x in line if x.isupper()]) / len(line)
    return (train['description'].map(str).map(upper_count_ratio),
            test['description'].map(str).map(upper_count_ratio))

# Number of upper case letters in description
def desc_num_count(train, test):
    def num_count(line):
        return len(RE_NUMBER.findall(line))
    return (train['description'].map(str).map(num_count),
            test['description'].map(str).map(num_count))

# Ratio of upper case letters in description
def desc_num_count_ratio(train, test):
    def num_count_ratio(line):
        if len(line) == 0:
            return 0
        return len(RE_NUMBER.findall(line)) / len(line)
    return (train['description'].map(str).map(num_count_ratio),
            test['description'].map(str).map(num_count_ratio))

# Number of unique words in description
def desc_uniq_wc(train, test):
    def uniq_wc(line):
        word_list = _normalize_text_word_list(line)
        return len(set(word_list))
    return (train['description'].map(str).map(uniq_wc),
            test['description'].map(str).map(uniq_wc))

# Ratio of unique words in description
def desc_uniq_wc_ratio(train, test):
    def uniq_wc_ratio(line):
        word_list = _normalize_text_word_list(line)
        if len(word_list) == 0:
            return 0
        return len(set(word_list)) / len(word_list)
    return (train['description'].map(str).map(uniq_wc_ratio),
            test['description'].map(str).map(uniq_wc_ratio))

# Utility functions

# Encode data from 0 to N, nan will be encoded as -1. If nan need to
# stay unchanged, need to transform the result.
def _encode_column(train, test, col):
    data = train[col].append(test[col])
    codes = data.astype('category').cat.codes
    # return train_code, test_code in pd.Series format.
    return pd.Series(codes[ : train.shape[0]]), pd.Series(codes[train.shape[0] :])


# Number of listings per entry in the given column.
# This is treated as an attribute of the column, calculated with both train and test data.
def _counts(train, test, col):
    data = train[col].append(test[col])
    count_dict = data.value_counts().to_dict()
    return train[col].map(count_dict), test[col].map(count_dict)

# Number of listings per tuple of the given columns
def _multi_counts(train, test, cols):
    data = train[[*cols, 'item_id']].append(test[[*cols, 'item_id']])
    count_dict = data.groupby(cols).count()
    result = data[cols].merge(count_dict, how='left', left_on=cols, right_index=True)
    result = result['item_id'].rename('+'.join(cols) + '-counts', inplace=True)
    result.fillna(0, inplace=True)
    result = result.astype(int, copy=False)
    return result[ : train.shape[0]], result[train.shape[0] : ]

# Aggregates on the metrics with the agg_funcs over the dimensions.
# dimensions, metrics and agg_funcs are all string lists.
# Always returns a Dataframe
def _aggregate(train, test, dimensions, metrics, agg_funcs):
    cols = dimensions + metrics
    data = train[cols].append(test[cols])
    metrics_agg = data.groupby(dimensions).agg(agg_funcs)
    # By default, merge makes a copy of the dataframe.
    result = data[dimensions].merge(metrics_agg, how='left', left_on=dimensions, right_index=True)
    result.drop(dimensions, axis=1, inplace=True)
    # Renames the columns with dimensions to prevent conflict.
    result.rename(lambda x: '+'.join(dimensions) + '-' + '-'.join(x), axis=1, inplace=True)
    return result[ : train.shape[0]], result[train.shape[0] : ]


def _aggregate_active(
    train, test, train_active, test_active, dimensions, metrics, agg_funcs):
    cols = dimensions + metrics
    data_train = train[cols]
    data_test = test[cols]
    data = data_train.append([data_test, train_active[cols], test_active[cols]])
    # There are duplicate item_ids in train_active and test_active.
    data.drop_duplicate(['item_id'], inplace=True)
    metrics_agg = data.groupby(dimensions).agg(agg_funcs)

    # By default, merge makes a copy of the dataframe.
    result_train = data_train[dimensions].merge(metrics_agg, how='left', left_on=dimensions, right_index=True)
    result_train.drop(dimensions, axis=1, inplace=True)
    # Renames the columns with dimensions to prevent conflict.
    result_train.rename(lambda x: '+'.join(dimensions) + '-' + '-'.join(x), axis=1, inplace=True)

    result_test = data_test[dimensions].merge(metrics_agg, how='left', left_on=dimensions, right_index=True)
    result_test.drop(dimensions, axis=1, inplace=True)
    # Renames the columns with dimensions to prevent conflict.
    result_test.rename(lambda x: '+'.join(dimensions) + '-' + '-'.join(x), axis=1, inplace=True)

    return result_train, result_test

# def get_metric(dimension_series, metric, agg_func, metrics_dict):
#     lookup_dict = metrics_dict[(metric, agg_func)]
#     key = tuple(dimension_series.tolist())
#     # Rows with nan values in groupby columns are excluded, set their aggregate value to 0.
#     if key not in lookup_dict:
#         return 0
#     return lookup_dict[key]

def _normalize_text(text):
    text = text.lower().strip()
    for s in string.punctuation:
        text = text.replace(s, ' ')
    text = text.strip().split(' ')
    return u' '.join(x for x in text if len(x) > 1 and x not in stopwords)

def _normalize_text_word_list(text):
    text = text.lower().strip()
    for s in string.punctuation:
        text = text.replace(s, ' ')
    text = text.strip().split(' ')
    return [x for x in text if len(x) > 1 and x not in stopwords]

