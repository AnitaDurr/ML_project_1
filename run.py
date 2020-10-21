#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from implementations import *
from proj1_helpers import *

# load training data into class labels, feature matrix and event ids
DATA_TRAIN_PATH = 'data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


# Generate predictions and save ouput in csv format for submission
DATA_TEST_PATH = 'data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = 'data/predict.csv'
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
