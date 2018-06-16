import inspect

f = open('feature_map.py', 'w')
# Write comments
comments = """ # DO NOT EDIT MANUALLY. This file is auto-generated using feature_map_generator.py.
# A dictionary for features, feature generator use this map to get the function
# that generates each feature.
# The keys are unique names to identify each feature. Features will be stored in
# the pickle with the following name by the feature generator.
# Other code will use name to read the proper pickles.
"""
f.write(comments)
f.write('\n')

# Write import
f.write('import features\n')
f.write('import features_active\n')
f.write('\n')

# Generate feature map
import features
funcs = inspect.getmembers(features, inspect.isfunction)
feature_map = []
for name, func in funcs:
  if name.startswith('_'):
    # private helper methods.
    continue
  line = "'%s': features.%s," %(name, name)
  feature_map.append(line)

code = """feature_map = {
  %s
}
""" %'\n  '.join(feature_map)


f.write(code)
f.write('\n')

# Generate active feature map
import features_active
active_funcs = inspect.getmembers(features_active, inspect.isfunction)
feature_map_active = []
for name, func in active_funcs:
  if name.startswith('_'):
    # private helper methods.
    continue
  line = "'%s': features_active.%s," %(name, name)
  feature_map_active.append(line)

code = """feature_map_active = {
  %s
}
""" %'\n  '.join(feature_map_active)

f.write(code)

f.close()
