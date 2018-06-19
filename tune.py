# Tuning a model
# Sample usage:
#   python tune.py -c config_name  # to tune parameters for the given configuration
#   python tune.py -c config_name -t trials_file  # to continue tune parameters for
#                                                   the given config, starting from
#                                                   the given trials.

# If encounter "TypeError: 'generator' object is not subscriptable",
# try pip install --upgrade git+git://github.com/hyperopt/hyperopt.git
# see https://github.com/hyperopt/hyperopt/pull/319
import os
import pickle
import time
from datetime import datetime
from optparse import OptionParser

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe

from configs import config_map
from train import cross_validate, prepare_data

TRIALS_FOLDER = 'trials/'

def tune_single_model(parameter_space, config_name, max_evals, trials=None):
    # Prepare train data.
    X, y = prepare_data(parameter_space['features'], parameter_space['image_feature_folders'], test=False)
    def train_wrapper(params):
        cv_losses, cv_train_losses = cross_validate(params, X, y)
        # return an object to be recorded in hyperopt trials for future uses
        return {
            'loss': np.mean(cv_losses),
            'train_loss': np.mean(cv_train_losses),
            'status': STATUS_OK,
            'eval_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'params': params
        }

    if trials is None:
        trials = Trials()
    # tuning parameters
    t1 = time.time()
    timestamp = datetime.now().strftime("%m-%d_%H:%M:%S")
    best = fmin(train_wrapper, parameter_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    t2 = time.time()
    print('best trial get at round: ' + str(trials.best_trial['tid']))
    print('best loss: ' + str(trials.best_trial['result']['loss']))
    print(best)
    print(space_eval(parameter_space, best))
    print("time: %s" %((t2-t1) / 60))

    # save the experiment trials in a pickle
    if not os.path.exists(TRIALS_FOLDER):
        os.makedirs(TRIALS_FOLDER)
    # TODO: save tuning config when dump trials pickle.
    pickle.dump(trials, open("%s%s_%s" %(TRIALS_FOLDER, config_name, timestamp), "wb"))

    return trials

def main():
    parser = OptionParser()
    # Flag for input which configuration to use.
    parser.add_option('-c', '--config', action='store', type='string', dest='config_name', default='')
    # Trials of the existing tuning, if provide, the trial should be based on the configuration
    # given through the above flag (exact same model and parameter space).
    parser.add_option('-t', '--trials', action='store', type='string', dest='trials_file', default='')
    # Parses cmd line arguments
    options, _ = parser.parse_args()

    config = config_map[options.config_name]
    print('Using config: ', options.config_name)

    trials = None
    # If an existing tuning's trials is given, continues tuning it.
    if options.trials_file != '':
        trials_path = TRIALS_FOLDER + options.trials_file
        trials = pickle.load(open(trials_path, 'rb'))
        print('Using trials: %s' %trials_path)

    tune_params = config['tune_params']
    tune_single_model(
        tune_params['param_space'],
        options.config_name,
        tune_params['max_evals'],
        trials)

if __name__ == '__main__':
    main()
