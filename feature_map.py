 # DO NOT EDIT MANUALLY. This file is auto-generated using feature_map_generator.py.
# A dictionary for features, feature generator use this map to get the function
# that generates each feature.
# The keys are unique names to identify each feature. Features will be stored in
# the pickle with the following name by the feature generator.
# Other code will use name to read the proper pickles.

import features
import features_active

feature_map = {
  'activation_date': features.activation_date,
  'activation_weekday': features.activation_weekday,
  'category_name': features.category_name,
  'city': features.city,
  'desc_exclam_count': features.desc_exclam_count,
  'desc_len_norm': features.desc_len_norm,
  'desc_len_wc_norm_ratio': features.desc_len_wc_norm_ratio,
  'desc_letter_punc_ratio': features.desc_letter_punc_ratio,
  'desc_num_count': features.desc_num_count,
  'desc_num_count_ratio': features.desc_num_count_ratio,
  'desc_punc_count': features.desc_punc_count,
  'desc_uniq_wc': features.desc_uniq_wc,
  'desc_uniq_wc_ratio': features.desc_uniq_wc_ratio,
  'desc_upper_count': features.desc_upper_count,
  'desc_upper_count_ratio': features.desc_upper_count_ratio,
  'desc_wc': features.desc_wc,
  'desc_wc_norm': features.desc_wc_norm,
  'desc_wc_norm_ratio': features.desc_wc_norm_ratio,
  'desc_wc_punc_ratio': features.desc_wc_punc_ratio,
  'description': features.description,
  'has_description': features.has_description,
  'has_image': features.has_image,
  'has_one_param': features.has_one_param,
  'has_price': features.has_price,
  'image': features.image,
  'image_top_1': features.image_top_1,
  'item_id': features.item_id,
  'item_seq_number': features.item_seq_number,
  'listings_per_city_date': features.listings_per_city_date,
  'listings_per_user': features.listings_per_user,
  'log_price': features.log_price,
  'param_1': features.param_1,
  'param_2': features.param_2,
  'param_3': features.param_3,
  'parent_category_name': features.parent_category_name,
  'price': features.price,
  'price_city_date_mean_max': features.price_city_date_mean_max,
  'region': features.region,
  'title': features.title,
  'title_exclam_count': features.title_exclam_count,
  'title_len': features.title_len,
  'title_len_wc_ratio': features.title_len_wc_ratio,
  'title_letter_punc_ratio': features.title_letter_punc_ratio,
  'title_num_count': features.title_num_count,
  'title_num_count_ratio': features.title_num_count_ratio,
  'title_punc_count': features.title_punc_count,
  'title_uniq_wc': features.title_uniq_wc,
  'title_uniq_wc_ratio': features.title_uniq_wc_ratio,
  'title_upper_count': features.title_upper_count,
  'title_upper_count_ratio': features.title_upper_count_ratio,
  'title_wc': features.title_wc,
  'title_wc_punc_ratio': features.title_wc_punc_ratio,
  'user_id': features.user_id,
  'user_type': features.user_type,
}

feature_map_active = {
  'city_date_price_mean_max_active': features_active.city_date_price_mean_max_active,
  'parent_cat_price_mean_active': features_active.parent_cat_price_mean_active,
  'parent_cat_price_median_active': features_active.parent_cat_price_median_active,
  'parent_cat_price_std_active': features_active.parent_cat_price_std_active,
}
