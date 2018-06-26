from catboost import CatBoostRegressor

class CatBoost():
    def __init__(self, model_params = None, data_params = None):
        self.model_params = model_params
        self.data_params = data_params
        self.model = CatBoostRegressor(**self.model_params)


    def fit(self, X_train, y_train, X_val, y_val):
        cat_features = self.data_params['categorical_feature']
        # X_train.fillna(-999, inplace=True)
        dtrain = CatBoost.Pool(X_train, y_train, cat_features=list(cat_features), feature_names=X_train.columns)
        # X_train.fillna(-999, inplace=True)
        return self.model.fit(dtrain)

    def fit_predict(self, X_train, y_train):
        cat_features = self.data_params['categorical_feature']
        dtrain = CatBoost.Pool(X_train, y_train, cat_features=list(cat_features), feature_names=X_train.columns)
        # X_train.fillna(-999, inplace=True)
        return self.model.fit(dtrain)

    def predict(self, X):
        """ Predict on the given X, need to call fit first
            Returns:
                an array of the predict results, has the same rows as X.
        """
        # X.fillna(-999, inplace=True)
        return self.model.predict(X)

    def get_params(self):
        return self.model_params if self.model_params is not None else {}

    def get_features_importances(self):
        return None