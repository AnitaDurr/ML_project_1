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


mask_pos = (y == 1)

for i in range(len(feat_names)):
    for j in range(i + 1, len(feat_names)):
        u = tX[:, i][mask_pos]
        v = tX[:, j][mask_pos]
        plt.plot(u, v, 'r.')
        
        u = tX[:, i][~mask_pos]
        v = tX[:, j][~mask_pos]
        plt.plot(u, v, 'b.')
        
        plt.xlabel(feat_names[i])
        plt.ylabel(feat_names[j])
        plt.show()
    
    
