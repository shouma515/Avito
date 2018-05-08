# All the functions in this file assume takes the raw train and test dataframes,
# and output a pair of serieses or dataframes contain one or more features
# generated for train and test datasets.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Primary key.
def item_id(train_df, test_df):
    return train_df['item_id'], test_df['item_id']

def user_id(train_df, test_df):
    return encode_column(train_df, test_df, 'user_id')

def region(train_df, test_df):
    return encode_column(train_df, test_df, 'region')

def city(train_df, test_df):
    return encode_column(train_df, test_df, 'city')

def parent_category_name(train_df, test_df):
    return encode_column(train_df, test_df, 'parent_category_name')

def category_name(train_df, test_df):
    return encode_column(train_df, test_df, 'category_name')

def param_1(train_df, test_df):
    return encode_column(train_df, test_df, 'param_1')

def param_2(train_df, test_df):
    return encode_column(train_df, test_df, 'param_2')

def param_3(train_df, test_df):
    return encode_column(train_df, test_df, 'param_3')

def title(train_df, test_df):
    return train_df['title'], test_df['title']

def description(train_df, test_df):
    return train_df['description'], test_df['description']

def price(train_df, test_df):
    return train_df['price'], test_df['price']

def item_seq_number(train_df, test_df):
    return train_df['item_seq_number'], test_df['item_seq_number']

# TODO: some model does not take date time, consider how to deal with them.
def activation_date(train_df, test_df):
    return train_df['activation_date'], test_df['activation_date']

def user_type(train_df, test_df):
    return encode_column(train_df, test_df, 'user_type')

def image(train_df, test_df):
    return train_df['image'], test_df['image']

def image_top_1(train_df, test_df):
    return encode_column(train_df, test_df, 'image_top_1')

# Encode data from 0 to N, nan will be encoded as -1. If nan need to
# stay unchanged, need to transform the result.
def encode_column(train_df, test_df, col):
    data = train_df[col].append(test_df[col])
    codes = data.astype('category').cat.codes
    # return train_code, test_code in pd.Series format.
    return pd.Series(codes[ : train_df.shape[0]]), pd.Series(codes[train_df.shape[0] :])
