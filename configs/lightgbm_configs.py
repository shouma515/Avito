# Image feature will come after normal features
# When changing feature list, also need to change 'categorial_feature' to
# specify which columns are categorical.
# IMPORTANT! Always put categorical features first in the feature list, as some
# features contains more than one columns, it will be hard to figure out
# categorical feature index if we mix them.

from hyperopt import hp

lightgbm_config_feature_list = [
  # Categorical features:
  'activation_weekday',
  'category_name',
  'city',
  'image_top_1',
  'param_1',
  'param_2',
  'param_3',
  'parent_category_name',
  'region',
  'user_id',
  'user_type',
  # Numerical and boolean features:
  'desc_exclam_count',
  'desc_len_norm',
  'desc_len_wc_norm_ratio',
  'desc_letter_punc_ratio',
  'desc_num_count',
  'desc_num_count_ratio',
  'desc_punc_count',
  'desc_uniq_wc',
  'desc_uniq_wc_ratio',
  'desc_upper_count',
  'desc_upper_count_ratio',
  'desc_wc',
  'desc_wc_norm',
  'desc_wc_norm_ratio',
  'desc_wc_punc_ratio',
#   'has_description',
#   'has_image',
#   'has_one_param',
#   'has_price',
  'item_seq_number',
#   'item_seq_number_below_fifty',
#   'item_seq_number_below_five',
#   'item_seq_number_below_hundred',
#   'item_seq_number_below_ten',
#   'item_seq_number_below_thirty',
#   'item_seq_number_below_twenty',
  'item_seq_number_bucket',
#   'item_seq_number_is_one',
  'log_price',
  'price',
  'title_exclam_count',
  'title_len',
  'title_len_wc_ratio',
  'title_letter_punc_ratio',
  'title_num_count',
  'title_num_count_ratio',
  'title_punc_count',
  'title_uniq_wc',
  'title_uniq_wc_ratio',
  'title_upper_count',
  'title_upper_count_ratio',
  'title_wc',
  'title_wc_punc_ratio',
#   'cat_city_price_mean_active',
#   'cat_city_price_median_active',
#   'cat_city_price_norm_active',
#   'cat_city_price_std_active',
#   'cat_date_price_mean_active',
#   'cat_date_price_median_active',
#   'cat_date_price_norm_active',
#   'cat_date_price_std_active',
#   'cat_price_mean_active',
#   'cat_price_median_active',
#   'cat_price_norm_active',
#   'cat_price_std_active',
#   'city_date_price_mean_max_active',
#   'listings_per_cat_city_date',
#   'listings_per_cat_date',
#   'listings_per_city_date',
#   'listings_per_user',
#   'listings_per_user_cat',
#   'listings_per_user_cat_ratio',
#   'listings_per_user_pcat',
#   'listings_per_user_pcat_ratio',
#   'parent_cat_price_mean_active',
#   'parent_cat_price_median_active',
#   'parent_cat_price_norm_active',
#   'parent_cat_price_std_active',
#   'bow_desc_1',
#   'bow_title_1',
  'brutal_count_1_3',
  'brutal_price_avg_1_3',
  'embedding',
]

lightgbm_config = {

        'features': lightgbm_config_feature_list,
        'folds': 5,
        'image_feature_folders': ['image_features/ads_image_features'],
        'model': 'lightgbm',
        'model_params': {
            'bagging_fraction': 0.898898411669864,
            'bagging_freq': 10,
            'boosting_type': 'gbdt',
            'categorical_feature': '0,1,2,3,4,5,6,7,8,9,10',
            'learning_rate': 0.15124909294038838,
            'max_bin': 255,
            'max_depth': 20,
            'metric': 'rmse',
            'min_data': 500,
            'min_data_in_bin': 300,
            'min_hessian': 0.40763623871852805,
            'num_leaves': 300,
            'num_boost_round': 1000,
            'objective': 'regression',
            'sub_feature': 0.2295600655921562,
            # 'top_k': 50,
            'top_k': 50,
            'seed': 42,
            'verbose': -1
        },

    # 'features': lightgbm_config_feature_list,
    # 'image_feature_folders': ['image_features/ads_image_features'],
    # 'folds':2,
    # 'model': 'lightgbm',
    # 'model_params': {
    #     'categorical_feature': '0,1,2,3,4,5,6,7,8,9,10',
    #     'objective' : 'regression',
    #     'metric' : 'rmse',
    #     'num_leaves' : 32,
    #     'max_depth': 15,
    #     'learning_rate' ,
    #     'feature_fraction' ,
    #     'num_boost_round': 5000,
    #     # 'early_stopping_round': 500,
    #     'verbosity' : -1
    # },
    'tune_params': {
        'param_space': {
            'features': lightgbm_config_feature_list,
            'image_feature_folders': ['image_features/ads_image_features'],
            'model': 'lightgbm',
            'folds': 5,
            'model_params': {
                'max_depth': hp.choice('max_depth', [5, 10, 20]),
                'min_hessian': hp.loguniform('min_hessian', -3, 1),
                'bagging_fraction': hp.uniform('bagging_fraction', 0.3, 0.9),
                'bagging_freq': hp.choice('bagging_freq', [0, 10, 50, 100]),
                'sub_feature': hp.uniform('sub_feature', 0.1, 0.5),
                'top_k':  hp.choice('top_k', [20,50,100]),
                'max_bin': hp.choice('max_bin', [255, 350, 500]),
                'min_data_in_bin': hp.choice('min_data_in_bin', [30, 100, 300]),

                'learning_rate': hp.loguniform('learning_rate', -2, 0),
                'boosting_type': 'gbdt',
                'categorical_feature': '0,1,2,3,4,5,6,7,8,9,10',
                'objective': 'regression',
                # 'metric': hp.choice('metric', ['mae', 'mse']),
                'metric': 'rmse',
                'num_leaves': hp.choice('num_leaves', [100, 200, 300]),
                'min_data': hp.choice('min_data', [300, 500, 750, 1000]),
                'num_boost_round': hp.choice('num_boost_round', [500, 750, 1000]),
                'verbose': -1
            },
        },
        'max_evals': 35
    }
}
