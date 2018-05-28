# All the functions in this file assume takes the raw train and test dataframes,
# and output a pair of serieses or dataframes contain one or more features
# generated for train and test datasets.
# IMPORTANT: all the functions here should be pure functions, i.e. they should
#            not modify the input train or test dataframes.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Original fields

# Primary key.
def item_id(train, test):
    return train['item_id'], test['item_id']

def user_id(train, test):
    return encode_column(train, test, 'user_id')

def region(train, test):
    return encode_column(train, test, 'region')

def city(train, test):
    return encode_column(train, test, 'city')

def parent_category_name(train, test):
    return encode_column(train, test, 'parent_category_name')

def category_name(train, test):
    return encode_column(train, test, 'category_name')

def param_1(train, test):
    return encode_column(train, test, 'param_1')

def param_2(train, test):
    return encode_column(train, test, 'param_2')

def param_3(train, test):
    return encode_column(train, test, 'param_3')

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
    return encode_column(train, test, 'user_type')

def image(train, test):
    return train['image'], test['image']

def image_top_1(train, test):
    return encode_column(train, test, 'image_top_1')



# Generated fields

# Counts
# Number of listings per column entry. Calculated with both train and test data.
# Number of listings per user_id.
def listings_per_user(train, test):
    return counts(train, test, 'user_id')

def listings_per_city_date(train, test):
    return multi_counts(train, test, ['city', 'activation_date'])

# Aggregation over some dimensions
# TODO: figure out how to aggregate / count over multiple dimension
def price_city_date_mean_max(train, test):
    return aggregate(
        train, test, ['city', 'activation_date'], ['price'], ['mean', 'max'])

# Boolean features
# Actually this can be implied by image_top_1 != nan
def has_image(train, test):
    return train['image'].notnull(), test['image'].notnull()

# Length features
# Title length
def title_len(train, test):
    return train['title'].map(len), test['title'].map(len)

# Title word counts
def title_wc(train, test):
    return train['title'].map(str.split).map(len), test['title'].map(str.split).map(len)

# Description length
def desc_len(train, test):
    train_desc_len = train['description'].map(str).map(len)
    # Set missing description length to 0. Otherwise they will be 3 ('nan').
    train_desc_len[train['description'].isnull()] = 0

    test_desc_len = test['description'].map(str).map(len)
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


# Utility functions

# Encode data from 0 to N, nan will be encoded as -1. If nan need to
# stay unchanged, need to transform the result.
def encode_column(train, test, col):
    data = train[col].append(test[col])
    codes = data.astype('category').cat.codes
    # return train_code, test_code in pd.Series format.
    return pd.Series(codes[ : train.shape[0]]), pd.Series(codes[train.shape[0] :])


# Number of listings per entry in the given column.
# This is treated as an attribute of the column, calculated with both train and test data.
def counts(train, test, col):
    data = train[col].append(test[col])
    count_dict = data.value_counts().to_dict()
    return train[col].map(count_dict), test[col].map(count_dict)

# Number of listings per tuple of the given columns
def multi_counts(train, test, cols):
    data = train[[*cols, 'item_id']].append(test[[*cols, 'item_id']])
    count_dict = data.groupby(cols).agg(['count']).to_dict()
    result = data[cols].apply(get_metric, axis=1, args=('item_id', 'count', count_dict))
    return result[ : train.shape[0]], result[train.shape[0] : ]

def aggregate(train, test, dimensions, metrics, agg_funcs):
    cols = dimensions + metrics
    data = train[cols].append(test[cols])
    metrics_dict = data.groupby(dimensions).agg(agg_funcs).to_dict()
    features = []
    for metric in metrics:
        for agg_func in agg_funcs:
            metric_series = data[dimensions].apply(get_metric, axis=1, args=(metric, agg_func, metrics_dict))
            col_name = '_'.join(dimensions) + '-' + metric + '-' + agg_func
            metric_series.rename(col_name, inplace=True)
            features.append(metric_series)
    if len(features) == 1:
        # Returns the series if there's only one feature.
        result = features[0]
    else:
        result = pd.concat(features, axis=1)
    return result[ : train.shape[0]], result[train.shape[0] : ]

def get_metric(dimension_series, metric, agg_func, metrics_dict):
    lookup_dict = metrics_dict[(metric, agg_func)]
    key = tuple(dimension_series.tolist())
    # Rows with nan values in groupby columns are excluded, set their aggregate value to 0.
    if key not in lookup_dict:
        return 0
    return lookup_dict[key]
