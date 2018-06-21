from hyperopt import hp

catboost_config_feature_list = [
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
  # 'user_id',
  'user_type',
  'item_seq_number_bucket',
  # Numerical and boolean features:
  # 'desc_exclam_count',
  'desc_len_norm',
  'desc_len_wc_norm_ratio',
  'desc_letter_punc_ratio',
  'desc_num_count',
  'desc_num_count_ratio',
  # 'desc_punc_count',
  'desc_uniq_wc',
  'desc_uniq_wc_ratio',
  'desc_upper_count',
  'desc_upper_count_ratio',
  # 'desc_wc',
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
#   'item_seq_number_is_one',
  'log_price',
  'price',
  # 'title_exclam_count',
  'title_len',
  'title_len_wc_ratio',
  'title_letter_punc_ratio',
  'title_num_count',
  'title_num_count_ratio',
  'title_punc_count',
  # 'title_uniq_wc',
  'title_uniq_wc_ratio',
  'title_upper_count',
  'title_upper_count_ratio',
  'title_wc',
  # 'title_wc_punc_ratio',
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

# Configuration
catboost_config = {
    'features': catboost_config_feature_list,
    'folds': 2,
    'image_feature_folders': ['image_features/ads_image_features'],
    'model': 'catboost',
    'categorical_feature': [0,1,4,5,7,8,9,10],
    'model_params': {
            'depth': 2,
            'eval_metric': 'RMSE',
            'iterations': 300,
            'l2_leaf_reg': 3,
            # 'learning_rate': 0.2118306337399935,
            'loss_function': 'RMSE',
            'random_seed': 42,
            'od_type': 'Iter',
            'od_wait': 15,
            'verbose': False,
    },
    'tune_params': {
        'param_space': {
            'features': catboost_config_feature_list,
            'folds': 2,
            'image_feature_folders': ['image_features/ads_image_features'],
            'model': 'catboost',
            'categorical_feature': [0,1,4,5,7,8,9,10],
            'model_params': {
                'iterations': hp.choice('iterations', [200, 300]),
                'learning_rate': hp.loguniform('learning_rate', -2, 0),
                'depth': hp.choice('depth', [3, 5, 7]),
                'l2_leaf_reg': hp.choice('l2_leaf_reg', [2, 3, 5]),
                'loss_function': 'RMSE',
                'eval_metric' : 'RMSE',
                'od_type': 'Iter',
                'od_wait': 15,
                'random_seed' : 42,
                'verbose': False,
            },
        },
        'max_evals': 10
    }
}