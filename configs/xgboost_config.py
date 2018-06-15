from hyperopt import hp

feature_list = [
    'user_id',
    'region',
    'city',
    'parent_category_name',
    'category_name',
    'param_1',
    'param_2',
    'param_3',
    'price',
    'item_seq_number',
    # 'activation_date',
    'user_type',
    'image_top_1',
    'has_image',
    'title_len',
    'title_wc',
    'desc_len',
    'desc_wc',
    'listings_per_user',
    'listings_per_city_date',
    'price_city_date_mean_max',
]
xgboost_config = {
    'features': feature_list,
    'folds':5,
    'model': 'xgboost',
    'model_params': {
      'alpha': 0.6,
      'colsample_bylevel': 0.7,
      'colsample_bytree': 0.7,
      'eta': 0.07901772316032044,
      'eval_metric': 'rmse',
      'gamma': 0.0018188912716341973,
      'lambda': 0.4,
      'max_depth': 4,
      'min_child_weight': 4.4156043204121,
      'objective': 'reg:linear',
      'silent': 1,
      'subsample': 0.6
    }
}
