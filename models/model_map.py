from .Lightgbm import Lightgbm

# A dictionary for models, other code use this map to get models.
# the keys are unique names to identify models.
model_map = {
    'lightgbm': Lightgbm
}