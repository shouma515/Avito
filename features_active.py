# All the functions in this file assume takes the raw train and test dataframes,
# and output a pair of serieses or dataframes contain one or more features
# generated for train and test datasets.
# IMPORTANT: all the functions here should be pure functions, i.e. they should
#            not modify the input train or test dataframes.
# All private function need to be start with _

import numpy as np
import pandas as pd

# Counts
# Number of listings per column entry. Calculated with both train and test data.
# Number of listings per user_id.
def listings_per_user(train, test, train_active, test_active):
    return _counts(train, test, train_active, test_active, 'user_id')

def listings_per_user_cat(train, test, train_active, test_active):
    return _multi_counts(train, test, train_active, test_active, ['user_id', 'category_name'])

def listings_per_user_cat_ratio(train, test, train_active, test_active):
    return _ratio_helper(
        listings_per_user_cat, listings_per_user,
        train, test, train_active, test_active,
        0)

def listings_per_user_pcat(train, test, train_active, test_active):
    return _multi_counts(train, test, train_active, test_active, ['user_id', 'parent_category_name'])

def listings_per_user_pcat_ratio(train, test, train_active, test_active):
    return _ratio_helper(
        listings_per_user_pcat, listings_per_user,
        train, test, train_active, test_active,
        0)

def listings_per_city_date(train, test, train_active, test_active):
    return _multi_counts(train, test, train_active, test_active, ['city', 'activation_date'])

def listings_per_cat_date(train, test, train_active, test_active):
    return _multi_counts(train, test, train_active, test_active, ['category_name', 'activation_date'])

def listings_per_cat_city_date(train, test, train_active, test_active):
    return _multi_counts(train, test, train_active, test_active, ['category_name', 'city', 'activation_date'])

# Price
def city_date_price_mean_max_active(train, test, train_active, test_active):
    return _aggregate(
        train, test, train_active, test_active, ['city', 'activation_date'], ['price'], ['mean', 'max'])

# Mean price of a parent category
def parent_cat_price_mean_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['parent_category_name'], ['price'], ['mean']))

# Median price of a parent category
def parent_cat_price_median_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['parent_category_name'], ['price'], ['median']))

# Std of price of a parent category
def parent_cat_price_std_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['parent_category_name'], ['price'], ['std']))

# Normalized price of an item within its parent category
def parent_cat_price_norm_active(train, test, train_active, test_active):
    train_mean, test_mean = parent_cat_price_mean_active(train, test, train_active, test_active)
    train_std, test_std = parent_cat_price_std_active(train, test, train_active, test_active)
    train_norm = _ratio(train['price'] - train_mean, train_std, 0)
    test_norm = _ratio(test['price'] - test_mean, test_std, 0)
    return train_norm, test_norm

# Mean price of a category
def cat_price_mean_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['category_name'], ['price'], ['mean']))

# Median price of a category
def cat_price_median_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['category_name'], ['price'], ['median']))

# Std of price of a category
def cat_price_std_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['category_name'], ['price'], ['std']))

# Normalized price of an item within its category
def cat_price_norm_active(train, test, train_active, test_active):
    train_mean, test_mean = cat_price_mean_active(train, test, train_active, test_active)
    train_std, test_std = cat_price_std_active(train, test, train_active, test_active)
    train_norm = _ratio(train['price'] - train_mean, train_std, 0)
    test_norm = _ratio(test['price'] - test_mean, test_std, 0)
    return train_norm, test_norm

# Mean price of a category and city
def cat_city_price_mean_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['category_name', 'region', 'city'], ['price'], ['mean']))

# Median price of a category and city
def cat_city_price_median_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['category_name', 'region', 'city'], ['price'], ['median']))

# Std of price of a category and city
def cat_city_price_std_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['category_name', 'region', 'city'], ['price'], ['std']))

# Normalized price of an item within its category and city
def cat_city_price_norm_active(train, test, train_active, test_active):
    train_mean, test_mean = cat_city_price_mean_active(train, test, train_active, test_active)
    train_std, test_std = cat_city_price_std_active(train, test, train_active, test_active)
    train_norm = _ratio(train['price'] - train_mean, train_std, 0)
    test_norm = _ratio(test['price'] - test_mean, test_std, 0)
    return train_norm, test_norm

# Mean price of a category and city
def cat_date_price_mean_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['category_name', 'activation_date'], ['price'], ['mean']))

# Median price of a category and city
def cat_date_price_median_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['category_name', 'activation_date'], ['price'], ['median']))

# Std of price of a category and city
def cat_date_price_std_active(train, test, train_active, test_active):
    return _retrieve_first_series(*_aggregate(
        train, test, train_active, test_active, ['category_name', 'activation_date'], ['price'], ['std']))

# Normalized price of an item within its category and city
def cat_date_price_norm_active(train, test, train_active, test_active):
    train_mean, test_mean = cat_date_price_mean_active(train, test, train_active, test_active)
    train_std, test_std = cat_date_price_std_active(train, test, train_active, test_active)
    train_norm = _ratio(train['price'] - train_mean, train_std, 0)
    test_norm = _ratio(test['price'] - test_mean, test_std, 0)
    return train_norm, test_norm

# Number of listings per entry in the given column.
# This is treated as an attribute of the column, calculated with both train and test data.
def _counts(train, test, train_active, test_active, col):
    cols = ['item_id', col]
    data = pd.concat([train[cols], test[cols], train_active[cols], test_active[cols]])
    # There are duplicate item_ids in train_active and test_active.
    data.drop_duplicates('item_id', inplace=True)
    # Drop item_id column as it is not used in grouping
    data = data[col]
    count_dict = data.value_counts().to_dict()
    return train[col].map(count_dict), test[col].map(count_dict)

# Number of listings per tuple of the given columns
def _multi_counts(train, test, train_active, test_active, cols):
    cols_id = ['item_id', *cols]
    data = pd.concat([train[cols_id], test[cols_id], train_active[cols_id], test_active[cols_id]])
    # There are duplicate item_ids in train_active and test_active.
    data.drop_duplicates('item_id', inplace=True)
    count_dict = data.groupby(cols).count()

    train_result = train[cols].merge(count_dict, how='left', left_on=cols, right_index=True)
    train_result = train_result['item_id'].rename('+'.join(cols) + '-counts', inplace=True)
    train_result.fillna(0, inplace=True)
    train_result = train_result.astype(int, copy=False)

    test_result = test[cols].merge(count_dict, how='left', left_on=cols, right_index=True)
    test_result = test_result['item_id'].rename('+'.join(cols) + '-counts', inplace=True)
    test_result.fillna(0, inplace=True)
    test_result = test_result.astype(int, copy=False)
    return train_result, test_result

# Utility functions
def _aggregate(
    train, test, train_active, test_active, dimensions, metrics, agg_funcs):
    # item_id column is needed to drop duplicate rows in train_active and
    # test_active
    cols = ['item_id'] + dimensions + metrics
    data_train = train[cols]
    data_test = test[cols]
    data = data_train.append([data_test, train_active[cols], test_active[cols]])
    # There are duplicate item_ids in train_active and test_active.
    data.drop_duplicates('item_id', inplace=True)
    # Drop item_id column as it is not used in grouping
    data.drop('item_id', axis=1, inplace=True)
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

# Retrieve first column as a series from train and test feature dataframes.
def _retrieve_first_series(train_feature_df, test_feature_df):
    return train_feature_df.ix[:,0], test_feature_df.ix[:,0]

# norminator and denominator have to be pandas series of same length.
# when present, default value is used when denominator is zero. (else, result
# will be inf, no error will be report)
# when present, fill na value is used to replace nan in result.
def _ratio(nominator, denominator, default_value=None, fillna_value=None):
    result = nominator / denominator
    if default_value is not None:
        # The row where cond is True (i.e. denominator not zero) is NOT changed.
        result.where(denominator != 0, default_value, inplace=True)
    if fillna_value is not None:
        result.where(result.notnull(), fillna_value)
    return result

# f1 and f2 are functions that calculate feature series. (No support for
# single column dataframe now).
# functions that returns feature dataframes cannot be used here.
# returns train and test feature series of f1/f2
def _ratio_helper(f1, f2,
        train, test, train_active, test_active,
        default_value=None, fillna_value=None):
    train_norm, test_norm = f1(train, test, train_active, test_active)
    train_denorm, test_denorm = f2(train, test, train_active, test_active)
    return (_ratio(train_norm, train_denorm, default_value, fillna_value),
            _ratio(test_norm, test_denorm, default_value, fillna_value))
