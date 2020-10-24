'''A script to select the best hyperparameters for each method'''

import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from cross_validation import *
from helpers import *
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

def cv_ls_gd_sgd(hprange, loss_tr_gd, loss_te_gd, loss_tr_sgd, loss_te_sgd):
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

# will be commented as this script will be called from run.py where this global variable is defined
DATA_TRAIN_PATH = 'data/train.csv'
print('===LOADING DATA===')
y, x, ids = load_clean_data(DATA_TRAIN_PATH)

print('===TUNE HYPERPARAMETERS===')

seed = 7
k_fold = 4
w_initial = np.zeros(x.shape[1])
max_iters = 1000

### LEAST SQUARE GD AND SGD
gammas = np.logspace(-10, -1, num=20)

print("[least square GD]", end=" ")


t1 = time.time()
best_GD_gamma, loss_tr_gd, loss_te_gd = tune_hyperparam(y, x, k_fold, seed, hprange=gammas, method=least_squares_GD, args=[w_initial, max_iters], compute_loss=compute_mse)
t2 = time.time()
print("time:", t2 - t1, "best gamma:", best_GD_gamma)
lsGD_args = [w_initial, max_iters, best_GD_gamma]

print("[least square SGD]", end=" ")

t1 = time.time()
best_SGD_gamma, loss_tr_sgd, loss_te_sgd = tune_hyperparam(y, x, k_fold, seed, hprange=gammas, method=least_squares_SGD, args=[w_initial, max_iters], compute_loss=compute_mse)
t2 = time.time()
print("time:", t2 - t1, "best gamma:", best_SGD_gamma)
lsSGD_args = [w_initial, max_iters, best_SGD_gamma]

# plot
cv_ls_gd_sgd(gammas, loss_tr_gd, loss_te_gd, loss_tr_sgd, loss_te_sgd)

### LEAST SQUARE RIDGE REGRESSSION

lambdas = np.arange(0, 10, 0.1)

print("[Ridge Regression]", end=" ")

t1 = time.time()
best_RR_lambda, loss_tr_rr, loss_te_rr = tune_hyperparam(y, x, k_fold, seed, hprange=lambdas, method=ridge_regression, args = [], compute_loss=compute_rmse)
t2 = time.time()
print("time:", t2 - t1, "best lambda:", best_RR_lambda)
rr_args = [best_RR_lambda]

#plot
cv_ls_rr(lambdas, loss_tr, loss_te)




### LOGISTIC REGRESSION



