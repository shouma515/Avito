from train import prepare_data
from sklearn.model_selection import KFold
import os
import time
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from configs import config_map

TEST_SIZE = 508438
SUBMISSION_FOLDER = 'submissions/'

def ensemble_for_lgb(config, X, y, X_test):
    categorical_feature = config['categorical_feature']
    model_params = config['model_params']
    file_code = len(X.columns)

    # Cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    predicts = []
    test_preds = []
    val_errors = []
    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        print('Fold ', i)

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val,y_val = X.iloc[val_index], y.iloc[val_index]

        X_train = X_train.drop('item_id', axis=1)
        val_id = X_val['item_id']
        X_val = X_val.drop('item_id', axis=1)
        # debug info
        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)

        temp_path = 'data/lgb_ens_%d_%d.csv' %(file_code, i)
        temp_path_binary = 'data/lgb_ens_%d_%d.bin' %(file_code, i)
        if not os.path.isfile(temp_path):
            t_start = time.time()
            print('save', temp_path)
            X_train.to_csv(temp_path, header=False)
            t_finish = time.time()
            print('Save csv time: ', (t_finish - t_start) / 60)
        if not os.path.isfile(temp_path_binary):
            d_train = lgb.Dataset(temp_path, label=y_train, feature_name=list(X_train.columns), categorical_feature=categorical_feature, free_raw_data=False)
            print('Save binary: ', temp_path_binary)
            d_train.save_binary(temp_path_binary)
        d_train = lgb.Dataset(temp_path_binary, feature_name=list(X_train.columns), categorical_feature=categorical_feature, free_raw_data=False)
        d_val = lgb.Dataset(X_val, label=y_val, feature_name=list(X_train.columns), categorical_feature=categorical_feature, free_raw_data=False)
        model = lgb.train(model_params.copy(), d_train, valid_sets=[d_train, d_val], valid_names=['train','valid'])
        predict = model.predict(X_val)
        np.clip(predict, 0, 1, out=predict)

        test_pred = model.predict(X_test)
        np.clip(test_pred, 0, 1, out=test_pred)
        test_preds.append(test_pred)

        val_error = math.sqrt(mean_squared_error(y_val, predict))
        print('validate caculated: %f' %val_error)
        val_errors.append(val_error)

        predict = pd.Series(predict, name='predict')
        val_id = val_id.reset_index(drop=True)
        predict = pd.concat([val_id, predict], axis=1)
        predicts.append(predict)
    return predicts, val_errors, test_preds

config = config_map['lightgbm_config']
ENSEMBLE_FOLDER = 'ensemble/'

X, y = prepare_data(config['features'] + ['item_id'], config['image_feature_folders'], test=False)
X_test, _ = prepare_data(config['features'], config['image_feature_folders'], test=True)
predicts, val_errors, test_preds = ensemble_for_lgb(config, X, y, X_test)

print('Avg validation error: %f' %(np.mean(val_errors)))
# for predict in predicts:
#     print(predict.shape)
# ensembled = pd.concat(predicts)
# print(ensembled.shape)
# assert(ensembled.shape == (len(X), 2))

# if not os.path.exists(ENSEMBLE_FOLDER):
#         os.makedirs(ENSEMBLE_FOLDER)
# ensembled.to_csv(ENSEMBLE_FOLDER+'lgb.csv', index=False)

pred_sum = test_preds[0] + test_preds[1] + test_preds[2] + test_preds[3] + test_preds[4]
prediction = pred_sum / 5
assert(len(prediction) == TEST_SIZE)

submission = pd.read_csv('data/sample_submission.csv')
# Sample submission file and test dataset has the same item_id
# in the same order.
submission['deal_probability'] = prediction

submission.to_csv(
    '%s%s_%s.csv' %(SUBMISSION_FOLDER, 'lightgbm', '5_folds'),
    index=False
)




