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
  'has_description',
  'has_image',
  'has_one_param',
  'has_price',
  'item_seq_number',
  'listings_per_city_date',
  'listings_per_user',
  'log_price',
  'price',
  'price_city_date_mean_max',
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
  'city_date_price_mean_max_active',
  'parent_cat_price_mean_active',
  'parent_cat_price_median_active',
  'parent_cat_price_std_active',
]

lightgbm_config = {
    'features': lightgbm_config_feature_list,
    'image_feature_folders': ['image_features/ads_image_features'],
    'folds':5,
    'model': 'lightgbm',
    'model_params': {
        'boosting_type': 'gbdt',
        'categorical_feature': '0,1,2,3,4,5,6,7,8,9,10',
        'learning_rate': 0.08,
        'max_bin': 110,
        'metric': 'mse',
        'min_data': 250,
        'min_hessian': 0.1,
        'num_boost_round': 750,
        'num_leaves': 75,
        'objective': 'regression',
        'sub_feature': 0.8,
        'verbose': -1
    },
    'tune_params': {
        'param_space': {
            'features': lightgbm_config_feature_list,
            'image_feature_folders': ['image_features/ads_image_features'],
            'model': 'lightgbm',
            'folds': 5,
            'model_params': {
                'learning_rate': hp.loguniform('learning_rate', -2, 0),
                'boosting_type': 'gbdt',
                'categorical_feature': '0,1,2,3,4,5,6,7,8,9,10',
                'objective': 'regression',
                # 'metric': hp.choice('metric', ['mae', 'mse']),
                'metric': 'mse',               
                'sub_feature': hp.uniform('sub_feature', 0.3, 0.7),
                'num_leaves': hp.choice('num_leaves', list(range(10, 151, 15))),
                'min_data': hp.choice('min_data', list(range(150, 301, 15))),
                'min_hessian': hp.loguniform('min_hessian', -3, 1),
                'num_boost_round': hp.choice('num_boost_round', [200, 300, 500]),
                'max_bin': hp.choice('max_bin', list(range(50, 151, 10))),
                # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                # 'bagging_freq': hp.choice('bagging_freq', list(range(0, 100, 10))),
                'verbose': -1
            },
        },
        'max_evals': 15
    }
}
