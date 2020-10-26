#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from implementations import *
from helpers import *
from compare_method import * 		# this will execute the compare_method script

# best_method and best_args should have been selected in compare_method script

weights, loss = best_method(y, tx, *best_args)

# Generate predictions and save ouput in csv format for submission
print('===LOADING TEST DATA===')
DATA_TEST_PATH = 'data/test.csv'
_, tx_test, ids_test = load_clean_data(DATA_TEST_PATH, feature_selection=False)
tx_test = np.delete(tx_test, features_to_remove, axis=1)

print('===PREDICTION===')
OUTPUT_PATH = 'data/predict.csv'
if best_method in [logistic_regression, reg_logistic_regression]:
    y_pred = predict_labels(weights, tx, is_LR=True, submission=True)
else:
    y_pred = predict_labels(weights, tx_test, is_LR=False, submission=True)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print('===SUBMISSION FILE CREATED===')
