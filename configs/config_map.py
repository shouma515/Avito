from .lightgbm_configs import lightgbm_config
from .lightgbm_configs_test import lightgbm_config_test

from .xgboost_config import xgboost_config

# A dictionary for configurations, other code use this map to get configs.
# the keys are unique names to identify configurations.
config_map = {
    'lightgbm_config': lightgbm_config,
    'xgboost_config': xgboost_config,
    'lightgbm_configs_test': lightgbm_config_test
}
