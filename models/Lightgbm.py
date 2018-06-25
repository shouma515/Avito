import os.path
import time

import lightgbm as lgb


class Lightgbm():
    def __init__(self, model_params = None, data_params = None):
        # TODO(hzn): check params one by one
        if model_params is None:
            model_params = {}
            model_params['learning_rate'] = 0.002
            model_params['boosting_type'] = 'gbdt'
            model_params['application'] = 'regression_l2'
            # model_params['sub_feature'] = 0.5
            model_params['num_leaves'] = 60
            model_params['min_data'] = 500
            model_params['min_hessian'] = 1
            model_params['num_boost_round'] = 500
            model_params['verbose'] = -1
        if data_params is None:
            data_params = {}
            data_params['test_size'] = 0.25
            data_params['random_state'] = 42
        self.model_params = model_params
        self.data_params = data_params
        self.model=None

    def fit(self, X_train, y_train):
        categorical_feature = self.data_params['categorical_feature']
        hash_code = len(X_train.columns)
        temp_path = 'data/lgb_dwsp_%d.csv' %hash_code
        temp_path_binary = 'data/lgb_dwsp_%d.bin' %hash_code
        print(X_train.shape)
        if not os.path.isfile(temp_path):
            t_start = time.time()
            print('save', temp_path)
            X_train.to_csv(temp_path, header=False)
            t_finish = time.time()
            print('Save csv time: ', (t_finish - t_start) / 60)
        if not os.path.isfile(temp_path_binary):
            X_y = lgb.Dataset(temp_path, label=y_train, feature_name=list(X_train.columns), categorical_feature=categorical_feature, free_raw_data=False)
            X_y.save_binary(temp_path_binary)
        d_train = lgb.Dataset(temp_path_binary, categorical_feature=categorical_feature, free_raw_data=False)

        self.model = lgb.train(self.model_params, d_train)

    def predict(self, X):
        """ Predict on the given X, need to call fit first
            Returns:
                an array of the predict results, has the same rows as X.
        """
        return self.model.predict(X)

    def get_params(self):
        return self.model_params if self.model_params is not None else {}

    def get_features_importances(self):
        return None
