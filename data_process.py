import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *

DATA_TRAIN_PATH = 'data/train.csv'

def pearson(X, Y):
		return np.cov(X, Y)[0, 1] / (np.std(X) * np.std(Y))

def standardize(x):
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x

def load_clean_data():
	y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

	### missing handling data : replace by mean of the observed data
	no_val = -999.0
	for j in range(tX.shape[1]):
		mean = np.mean(tX[:, j] != no_val)
		tX[:, j][tX[:, j] == no_val] = mean


	### normalize data
	tX = standardize(tX)


	### remove features uncorrelated to labels
	nb_features = tX.shape[1]
	cor_matrix = np.zeros(nb_features)
	for j in range(nb_features):
		cor = pearson(tX[:, j], y)
		cor_matrix[j] = cor

	ind = np.where(np.absolute(cor_matrix) > 0.03)
	y, tX, ids = y[ind], tX[ind], ids[ind]

	return y, tX, ids
