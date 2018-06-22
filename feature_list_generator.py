import os
import json

def is_feature(filename):
    not_feature = ['df', 'test']
    return all(x not in filename for x in not_feature)

pickles = os.listdir('pickles')
features = list(filter(is_feature, pickles))
features.sort()
with open('feature_list', 'w') as f:
    json.dump(features, f, indent=4) 