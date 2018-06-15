import xgboost as xgb

class XGBoost():
    def __init__(self, model_params = None, train_params = None):
        # TODO(hzn): check params one by one
        if model_params is None:
            model_params = {
                'eta': 0.037,
                'max_depth': 5,
                'subsample': 0.80,
                'objective': 'reg:linear',
                'eval_metric': 'mae',
                'lambda': 0.8,
                'alpha': 0.4,
                # 'base_score': y_mean,
                'silent': 1,
                # 'booster': 'gblinear'
            }
        if train_params is None:
            train_params = {
                'num_boost_round': 250
            }
        self.model_params = model_params
        self.train_params = train_params
        self.model = None

    def fit(self, X_train, y_train):
        d_train = xgb.DMatrix(X_train, y_train)
        self.model = xgb.train(dict(self.model_params, silent=1), d_train,
            **self.train_params)

    def predict(self, X):
        """ Predict on the given X, need to call fit first
            Returns:
                an array of the predict results, has the same rows as X.
        """
        data = xgb.DMatrix(X)
        return self.model.predict(data)

    def get_params(self):
        return self.model_params if self.model_params is not None else {}

    def get_features_importances(self):
        return self.model.get_fscore() if self.model is not None else None
