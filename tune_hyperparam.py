'''
A script to select the best hyperparameters for each method and store them in a file
'''

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from helpers import *
from cross_validation import *
import time


def tune_hyperparam(y, x, k_fold, seed, hprange, method, args, compute_loss):
	"""
	For the given method, computes the cross-validation loss of every hyperparameter
	in hprange and selects the best one.
	"""
	k_indices = build_k_indices(y, k_fold, seed)

	# define lists to store the loss of training data and test data
	loss_tr = []
	loss_te = []

	# cross validation
	for hp in hprange:
		# compute the cv_losses
		cv_tr, cv_te = cv_loss(y, x, k_indices, method, args + [hp], compute_loss)

		loss_tr.append(cv_tr)
		loss_te.append(cv_te)

	# keep the hyperparam giving best loss on the test set
	ind_hp_opt = np.argmin(loss_te)
	best_hp = hprange[ind_hp_opt]

	return best_hp, loss_tr, loss_te

def cross_validation_visualization(hprange, loss_tr, loss_te, x_label, y_label, title):
	"""
	Visualization the curves of loss_tr and loss_te.
	"""
	plt.semilogx(hprange, loss_tr, marker=".", color='b', label='train error')
	plt.semilogx(hprange, loss_te, marker=".", color='r', label='test error')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend(loc=2)
	plt.grid(True)
	plt.savefig(title)

def cv_gd_sgd(hprange, loss_tr_gd, loss_te_gd, loss_tr_sgd, loss_te_sgd):
	plt.figure()
	plt.semilogx(hprange, loss_tr_gd, marker=".", color='b', label='GD train error')
	plt.semilogx(hprange, loss_te_gd, marker=".", color='c', label='GD test error')
	plt.semilogx(hprange, loss_tr_sgd, marker=".", color='r', label='SGD train error')
	plt.semilogx(hprange, loss_te_sgd, marker=".", color='m', label='SGD test error')
	plt.xlabel("gamma")
	plt.ylabel("mse loss")
	title = "Tuning the gamma for the least square GD and SGD"
	plt.title(title)
	plt.legend(loc=2)
	plt.grid(True)
	plt.savefig("GD_SGD")

### SCRIPT

print('===LOADING DATA===')
y, x, ids = load_clean_data(DATA_TRAIN_PATH)

print('===TUNE HYPERPARAMETERS===')

TUNE_PATH = 'best_param.txt'
file = open(TUNE_PATH, "a+")

k_fold = 5
seed = 7

### LEAST SQUARE GD AND SGD
gammas = np.logspace(-10, -1, num=50)
w_initial = np.zeros(x.shape[1])
max_iters = 1000

print("[least square GD]", end=" ")

t1 = time.time()
best_GD_gamma, loss_tr_gd, loss_te_gd = tune_hyperparam(y, x, k_fold, seed, hprange=gammas, method=least_squares_GD, args=[w_initial, max_iters], compute_loss=compute_mse)
t2 = time.time()
print("time:", t2 - t1, "best gamma:", best_GD_gamma)

print("[least square SGD]", end=" ")
t1 = time.time()
best_SGD_gamma, loss_tr_sgd, loss_te_sgd = tune_hyperparam(y, x, k_fold, seed, hprange=gammas, method=least_squares_SGD, args=[w_initial, max_iters], compute_loss=compute_mse)
t2 = time.time()
print("time:", t2 - t1, "best gamma:", best_SGD_gamma)

# store bests
file.write("least square GD = {}\n".format({"gamma" : best_GD_gamma}))
file.write("least square SGD = {}\n".format({"gamma" : best_SGD_gamma}))


# ploTS
cross_validation_visualization(gammas, loss_tr_gd, loss_te_gd, "gamma", "mse loss", "least square GD cross validation")
cross_validation_visualization(gammas, loss_tr_sgd, loss_te_sgd, "gamma", "mse loss", "least square SGD cross validation")
cv_gd_sgd(gammas, loss_tr_gd, loss_te_gd, loss_tr_sgd, loss_te_sgd)

### LEAST SQUARE NORMAL EQUATION AND RIDGE REGRESSSION

### LOGISTIC REGRESSION


file.close()


