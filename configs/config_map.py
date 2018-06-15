from .lightgbm_configs import lightgbm_config
from .xgboost_config import xgboost_config

# A dictionary for configurations, other code use this map to get configs.
# the keys are unique names to identify configurations.
config_map = {
    'lightgbm_config': lightgbm_config,
    'xgboost_config': xgboost_config
}
