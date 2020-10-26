#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from implementations import *
from helpers import *
from compare_method import *

weights = best_method(y, tx, best_args)

# Generate predictions and save ouput in csv format for submission
DATA_TEST_PATH = 'data/test.csv'
OUTPUT_PATH = 'data/predict.csv'
_, tx_test, ids_test = load_clean_data(DATA_TEST_PATH)
y_pred = predict_labels(weights, tx_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
