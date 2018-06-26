# All the functions in this file assume takes the raw train and test dataframes,
# and output a pair of serieses or dataframes contain one or more features
# generated for train and test datasets.
# IMPORTANT: all the functions here should be pure functions, i.e. they should
#            not modify the input train or test dataframes.
# All private function need to be start with _

import numpy as np
import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA


# Global variables
STOPWORDS = {x: 1 for x in stopwords.words('russian')}
RE_PUNCTUATION = re.compile('|'.join([re.escape(x) for x in string.punctuation]))
RE_NUMBER = re.compile(r'\d+')



# Original fields

# Primary key.
def item_id(train, test):
    return train['item_id'], test['item_id']

def user_id(train, test):
    return _encode_column(train, test, 'user_id')

def region(train, test):
    return _encode_column(train, test, 'region')

def city(train, test):
    # City need to be scoped by region to be unique.
    train_city = train['region'] + train['city']
    test_city = test['region'] + test['city']
    data = train_city.append(test_city)
    codes = data.astype('category').cat.codes
    # return train_code, test_code in pd.Series format.
    return pd.Series(codes[ : train.shape[0]]), pd.Series(codes[train.shape[0] :])

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

# Activation date
def activation_weekday(train, test):
    return train['activation_date'].dt.weekday, test['activation_date'].dt.weekday

# Generated fields

# Boolean features
# Actually this can be implied by image_top_1 != nan
def has_image(train, test):
    return train['image'].notnull(), test['image'].notnull()

# Has at list one param
def has_one_param(train, test):
    return (train['param_1'].notnull() | train['param_2'].notnull() | train['param_3'].notnull(),
            test['param_1'].notnull() | test['param_2'].notnull() | test['param_3'].notnull())

def has_description(train, test):
    return train['description'].notnull(), test['description'].notnull()

# https://www.kaggle.com/c/avito-demand-prediction/discussion/56531
# this post says that ambiguity in infomation causes user to click more on ads,
# thus lead to a higher deal probability, which is an interesting point.
def has_price(train, test):
    return train['price'].notnull(), test['price'].notnull()

# Price features
def log_price(train, test):
    return np.log(train['price'] + 0.001), np.log(test['price'] + 0.001)

# Text meta features
# Length features
# Title length
def title_len(train, test):
    return train['title'].map(len), test['title'].map(len)

# Title word counts
def title_wc(train, test):
    return train['title'].map(str.split).map(len), test['title'].map(str.split).map(len)

# Title letters per word
def title_len_wc_ratio(train, test):
    def len_wc_ratio(line):
        word_list = line.split()
        if len(word_list) == 0:
            return 0
        line_len = sum([len(w) for w in word_list])
        return line_len / len(word_list)
    return train['title'].map(len_wc_ratio), test['title'].map(len_wc_ratio)

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

# Number of punctuations in title.
def title_punc_count(train, test):
    def punc_count(line):
        return len(RE_PUNCTUATION.findall(line))
    return train['title'].map(punc_count), test['title'].map(punc_count)

# Ratio of letters to punctuations in title
def title_letter_punc_ratio(train, test):
    return _ratio_helper(title_len, title_punc_count, train, test, 99999.9)

# Ratio of words to punctuations in title
def title_wc_punc_ratio(train, test):
    return _ratio_helper(title_wc, title_punc_count, train, test, 99999.9)

# Description length
def desc_len_norm(train, test):
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

    return train_desc_wc_ratio, test_desc_wc_ratio

# Description letters per word, normalized
def desc_len_wc_norm_ratio(train, test):
    def len_wc_norm_ratio(line):
        word_list = _normalize_text_word_list(line)
        if len(word_list) == 0:
            return 0
        line_len = sum([len(w) for w in word_list])
        return line_len / len(word_list)

    train_desc_len_wc_norm_ratio = train['description'].map(str).map(len_wc_norm_ratio)
    # Set missing description length to 0. Otherwise they will be 1 (['nan']).
    train_desc_len_wc_norm_ratio[train['description'].isnull()] = 0

    test_desc_len_wc_norm_ratio = test['description'].map(str).map(len_wc_norm_ratio)
    test_desc_len_wc_norm_ratio[test['description'].isnull()] = 0

    return train_desc_len_wc_norm_ratio, test_desc_len_wc_norm_ratio

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

# Ratio of letters to punctuations in description.
def desc_letter_punc_ratio(train, test):
    return _ratio_helper(desc_len_norm, desc_punc_count, train, test, 99999.9)

# Ratio of words to punctuations in description
def desc_wc_punc_ratio(train, test):
    return _ratio_helper(desc_wc_norm, desc_punc_count, train, test, 99999.9)

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

# Number of numbers in description
def desc_num_count(train, test):
    def num_count(line):
        return len(RE_NUMBER.findall(line))
    return (train['description'].map(str).map(num_count),
            test['description'].map(str).map(num_count))

# Ratio of numbers in description
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
    train_desc_uniq_wc = train['description'].map(str).map(uniq_wc)
    # Set missing description length to 0. Otherwise they will be 1 (['nan']).
    train_desc_uniq_wc[train['description'].isnull()] = 0

    test_desc_uniq_wc = test['description'].map(str).map(uniq_wc)
    test_desc_uniq_wc[test['description'].isnull()] = 0
    return train_desc_uniq_wc, test_desc_uniq_wc

# Ratio of unique words in description
def desc_uniq_wc_ratio(train, test):
    return _ratio_helper(desc_uniq_wc, desc_len_norm, train, test, 0)


# Item seq number
def item_seq_number_bucket(train, test):
    data = train['item_seq_number'].append(test['item_seq_number'])
    buckets = pd.cut(data, [0, 1, 2, 3, 4, 10, 14, 20, 30, 40, 50, 60, 70, 80, 90, 120, 200, 300, 400, 500, 600, 700, 1700, data.max()])
    codes = buckets.astype('category').cat.codes
    # return train_code, test_code in pd.Series format.
    return pd.Series(codes[ : train.shape[0]]), pd.Series(codes[train.shape[0] :])

def item_seq_number_is_one(train, test):
    return train['item_seq_number'] == 1, test['item_seq_number'] == 1

def item_seq_number_below_five(train, test):
    return train['item_seq_number'] < 5, test['item_seq_number'] < 5

def item_seq_number_below_ten(train, test):
    return train['item_seq_number'] < 10, test['item_seq_number'] < 10

def item_seq_number_below_twenty(train, test):
    return train['item_seq_number'] < 20, test['item_seq_number'] < 20

def item_seq_number_below_thirty(train, test):
    return train['item_seq_number'] < 30, test['item_seq_number'] < 30

def item_seq_number_below_fifty(train, test):
    return train['item_seq_number'] < 50, test['item_seq_number'] < 50

def item_seq_number_below_hundred(train, test):
    return train['item_seq_number'] < 100, test['item_seq_number'] < 100

# BOW features
# def bow_title_1(train, test):
#     count_vectorizer_title = CountVectorizer(
#         stop_words=stopwords.words('russian'), lowercase=True,
#         max_df=0.7, max_features=150)
#     title_counts = count_vectorizer_title.fit_transform(train['title'].append(test['title']))

#     train_result = pd.DataFrame(title_counts[:len(train)].todense())
#     test_result =  pd.DataFrame(title_counts[len(train):].todense())
#     train_result.rename(lambda x: 'title_' + str(x), axis=1, inplace=True)
#     test_result.rename(lambda x: 'title_' + str(x), axis=1, inplace=True)

#     return train_result, test_result

# def bow_desc_1(train, test):
#     count_vectorizer_desc = TfidfVectorizer(
#         lowercase=True, ngram_range=(1, 2),
#         max_features=150)
#     train_d = train['description'].astype(str).map(_normalize_text_remove_digits)
#     test_d = test['description'].astype(str).map(_normalize_text_remove_digits)

#     desc_counts = count_vectorizer_desc.fit_transform(train_d.append(test_d))

#     train_result = pd.DataFrame(desc_counts[:len(train)].todense())
#     test_result =  pd.DataFrame(desc_counts[len(train):].todense())
#     train_result.rename(lambda x: 'desc_' + str(x), axis=1, inplace=True)
#     test_result.rename(lambda x: 'desc_' + str(x), axis=1, inplace=True)

#     return train_result, test_result

def price_item_seq_number_ratio(train, test):
    return _ratio_helper(price, item_seq_number, train, test, 0)

def log_item_seq_number(train, test):
    return np.log(train['item_seq_number']), np.log(test['item_seq_number'])

def price_log_item_seq_ratio(train, test):
    return _ratio_helper(price, log_item_seq_number, train, test, 0)

def log_price_log_item_seq_ratio(train, test):
    return _ratio_helper(log_price, log_item_seq_number, train, test, 0)

def _embedding_features(df, embeddings_index, pca_dim):
    # df = df_main[['title','description']].copy()
    stop_words = stopwords.words('russian')
    # fill blank
    df["txt"] = df["title"].fillna(' ') + " " + df["description"].fillna(' ')
    df["txt"] = df["txt"].str.lower()
    # remove punctuation
    df["txt"] = df["txt"].str.replace(r'[^\w\s]',' ')
    # remove stopwords
    print("removing stopwords....")
    df["txt"] = df["txt"].apply(lambda line: " ".join(word for word in line.split() if word not in stop_words))

    print("calculating mean embedding vecs.....")
    df['embedding'] = df['txt'].apply(lambda line: np.mean([embeddings_index[word] for word in line.split() if word in embeddings_index.keys()],axis=0))
    s = df['embedding']
    result = pd.DataFrame.from_items(zip(s.index, s.values)).T
    print("Start PCA to %d dimension....." %pca_dim)
    result.fillna(0, inplace=True)
    pca = PCA(n_components = pca_dim)
    result = pd.DataFrame(pca.fit_transform(result))
    result.rename(columns = lambda x: 'embedding_'+str(x), inplace=True)
    return result

def embedding(train, test):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    # original embedding matrix
    print("reading pretrained embedding weights....")
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('data/cc.ru.300.vec'))
    data = pd.concat([train[['title','description']], test[['title','description']]])
    embeddings = _embedding_features(data, embeddings_index, 100)
    return embeddings[:len(train)], embeddings[len(train):]

# From: https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm/output
def period_features(train, test):
    period_features = pd.read_csv('data/aggregated_features.csv')
    train = train.merge(period_features, 'left', 'user_id')
    test = test.merge(period_features, 'left', 'user_id')
    columns = ['avg_days_up_user', 'avg_times_up_user', 'n_user_items']
    return train[columns], test[columns]

def regional_income(train,test):
    '''a generic function that takes the train and test data and returns
    a new copy with external feature added'''
    # add feature regional gdp growth rate and income
    regional_economy=pd.read_csv('data/regional_economy.csv',index_col=0)
    columns = ['income2018', 'growth_rate']
    train = train.merge(regional_economy, 'left', left_on='region', right_index=True)
    test = test.merge(regional_economy, 'left', left_on='region', right_index=True)
    # train['regional_income']=train.region.apply(lambda x: regional_economy.loc[x,'income2018'])
    # test['regional_income']=test.region.apply(lambda x: regional_economy.loc[x,'income2018'])
    # add more feature

    return train[columns], test[columns]

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
# Always returns a DataFrame
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

def _retrieve_first_series(df):
    return df.ix[:,0]

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
    return u' '.join(x for x in text if len(x) > 1 and x not in STOPWORDS)

def _normalize_text_remove_digits(text):
    text = text.lower().strip()
    for s in string.punctuation:
        text = text.replace(s, ' ')
    text = text.strip().split(' ')
    return u' '.join(x for x in text if len(x) > 1 and x not in STOPWORDS)

def _normalize_text_word_list(text):
    text = text.lower().strip()
    for s in string.punctuation:
        text = text.replace(s, ' ')
    text = text.strip().split(' ')
    return [x for x in text if len(x) > 1 and (not x.isdigit()) and (x not in STOPWORDS)]

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
        result.where(result.notnull(), fillna_value, inplace=True)
    return result

# f1 and f2 are functions that calculate feature series.
# functions that returns feature dataframes cannot be used here.
# returns train and test feature series of f1/f2
def _ratio_helper(f1, f2, train, test, default_value=None, fillna_value=None):
    train_norm, test_norm = f1(train, test)
    train_denorm, test_denorm = f2(train, test)
    return (_ratio(train_norm, train_denorm, default_value, fillna_value),
            _ratio(test_norm, test_denorm, default_value, fillna_value))
