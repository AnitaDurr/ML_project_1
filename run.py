#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from implementations import *
from helpers import *

# load training data into class labels, feature matrix and event ids
# DATA_TRAIN_PATH = 'data/train.csv'
# print('===LOADING DATA===')
# y, tx, ids = load_clean_data(DATA_TRAIN_PATH)

from compare_method import * 		# this will execute the compare_method script

# best_method and best_args should have been selected in compare_method script

weights = best_method(y, tx, best_args)

# Generate predictions and save ouput in csv format for submission
DATA_TEST_PATH = 'data/test.csv'
_, tx_test, ids_test = load_clean_data(DATA_TEST_PATH)


OUTPUT_PATH = 'data/predict.csv'
y_pred = predict_labels(weights, tx_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
