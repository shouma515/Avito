from .CatBoost import CatBoost
from .Lightgbm import Lightgbm
from .XGBoost import XGBoost

# A dictionary for models, other code use this map to get models.
# the keys are unique names to identify models.
model_map = {
    'catboost': CatBoost,
    'lightgbm': Lightgbm,
    'xgboost': XGBoost
}
