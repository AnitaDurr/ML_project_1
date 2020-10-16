#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from proj1_helpers import *


# load training data into class labels, feature matrix and event ids
DATA_TRAIN_PATH = 'data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=True)


with open(DATA_TRAIN_PATH, 'r') as f:
    d_reader = csv.DictReader(f)

    #get fieldnames from DictReader object and store in list
    feat_names = d_reader.fieldnames[2:]


for i in range(len(feat_names)):
    plt.plot(tX[:, i], y, '.')
    plt.xlabel(feat_names[i])
    plt.ylabel('label')
    plt.show()