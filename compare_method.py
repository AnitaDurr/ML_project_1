'''A script that compares machine learning methods using for each of them
the best hyperparameters, and select the best model.'''

import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from helpers import *
from tune_hyperparam import *

print("===COMPARE METHODS===")

# add arguments for your functions (shoudl actually be variables defined at the enf of the hyperparam tuning)
methods = [least_squares_GD, least_squares_SGD, least_squares, ridge_regression, logistic_regression, reg_logistic_regression]
arguments = [lsGD_args, lsSGD_args, [], rr_args, None, None]

seed = 42
k_fold = 4
k_indices = build_k_indices(y, k_fold, seed)

criterions = np.zeros((len(methods), 5), dtype=object)
for i in range(4):
    method = methods[i]
    args = arguments[i]

    accuracy, precision, recall, f1_score = cv_criterions(y, x, k_indices, method, args)

    criterions[i] = np.array([method.__name__, accuracy, precision, recall, f1_score])
