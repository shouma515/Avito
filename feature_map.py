import features

# A dictionary for features, feature generator use this map to get the function
# that generates each feature.
# The keys are unique names to identify each feature. Features will be stored in
# the pickle with the following name by the feature generator.
# Other code will use name to read the proper pickles.
feature_map = {
    'item_id': features.item_id,
    'user_id': features.user_id,
    'region': features.region,
    'city': features.city,
    'parent_category_name': features.parent_category_name,
    'category_name': features.category_name,
    'param_1': features.param_1,
    'param_2': features.param_2,
    'param_3': features.param_3,
    'title': features.title,
    'description': features.description,
    'price': features.price,
    'item_seq_number': features.item_seq_number,
    'activation_date': features.activation_date,
    'user_type': features.user_type,
    'image': features.image,
    'image_top_1': features.image_top_1,
    'has_image': features.has_image,
    'title_len': features.title_len,
    'title_wc': features.title_wc,
    'desc_len': features.desc_len,
    'desc_wc': features.desc_wc,
    'listings_per_user': features.listings_per_user,
    'listings_per_city_date': features.listings_per_city_date,
    'price_city_date_mean_max': features.price_city_date_mean_max,
}
