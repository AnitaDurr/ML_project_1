#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from implementations import *
from helpers import *
from compare_method import *

# best_method and best_args should have been selected in compare_method script

weights, loss = best_method(y, tx, *best_args)

# Generate predictions and save ouput in csv format for submission
print('===LOADING TEST DATA===')
DATA_TEST_PATH = 'data/test.csv'
_, tx_test, ids_test = load_clean_data(DATA_TEST_PATH)


OUTPUT_PATH = 'data/predict.csv'
is_LR = False
if best_method in [logistic_regression, reg_logistic_regression]:
    y_pred = predict_labels(weights, tx, is_LR=True)
else:
    y_pred = predict_labels(weights, tx_test, is_LR=False)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
