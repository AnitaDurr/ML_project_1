'''A script to select the best hyperparameters for each method'''

import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from cross_validation import *
from helpers import *
import time
import seaborn as sns


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
	plt.savefig("GD_SGD.pdf")
	plt.close()

def cv_ls_rr(lambda_range, loss_tr, loss_te):
	plt.figure()
	plt.plot(lambda_range, loss_tr, marker = ".", color = 'b', label = 'Train error')
	plt.plot(lambda_range, loss_te, marker = ".", color = 'r', label = 'Test error')
	plt.xlabel("lambda")
	plt.ylabel("rmse loss")
	plt.title("Tuning lambda for Least Squares Ridge Regression")
	plt.legend(loc=2)
	plt.grid(True)
	plt.savefig("RR.pdf")
	plt.close()

def cv_gam_log(hprange, loss_tr, loss_te):
	plt.figure()
	plt.semilogx(hprange, loss_tr, marker=".", color='b', label='Train error')
	plt.semilogx(hprange, loss_te, marker=".", color='r', label='Test error')
	plt.xlabel("gamma")
	plt.ylabel("Normalised -log likelihood loss")
	plt.title("Tuning gamma for Logistic regression")
	plt.legend(loc=2)
	plt.grid(True)
	plt.savefig("Gamma_logistic.pdf")
	plt.close()

def cv_reg_log(hprange, loss_l0, loss_l10, loss_l20):
	plt.figure()
	plt.semilogx(hprange, loss_l0, marker=".", color='b', label='Lambda=0')
	plt.semilogx(hprange, loss_l10, marker=".", color='c', label='Lambda=10')
	plt.semilogx(hprange, loss_l20, marker=".", color='m', label='Lambda=20')
	plt.xlabel("gamma")
	plt.ylabel("Normalised -log likelihood loss")
	plt.title("Tuning lambda for Regularised logistic regression (CV test losses)")
	plt.legend(loc=2)
	plt.grid(True)
	plt.savefig("Lambda_reg_log.pdf")
	plt.close()

### SCRIPT

# will be commented as this script will be called from run.py where this global variable is defined DONE
DATA_TRAIN_PATH = 'data/train.csv'
print('===LOADING DATA===')
y, tx, ids = load_clean_data(DATA_TRAIN_PATH)

print('===TUNE HYPERPARAMETERS===')

seed = 7
k_fold = 4
w_initial = np.zeros(tx.shape[1])
max_iters = 1000

### LEAST SQUARE GD AND SGD
gammas = np.logspace(-10, -1, num=20)

print("[least square GD]", end=" ")


t1 = time.time()
best_GD_gamma, loss_tr_gd, loss_te_gd = tune_hyperparam(y, tx, k_fold, seed, hprange=gammas, method=least_squares_GD, args=[w_initial, max_iters], compute_loss=compute_mse)
t2 = time.time()
print("time:", t2 - t1, "best gamma:", best_GD_gamma)
lsGD_args = [w_initial, max_iters, best_GD_gamma]

print("[least square SGD]", end=" ")

t1 = time.time()
best_SGD_gamma, loss_tr_sgd, loss_te_sgd = tune_hyperparam(y, tx, k_fold, seed, hprange=gammas, method=least_squares_SGD, args=[w_initial, max_iters], compute_loss=compute_mse)
t2 = time.time()
print("time:", t2 - t1, "best gamma:", best_SGD_gamma)
lsSGD_args = [w_initial, max_iters, best_SGD_gamma]

# plot
cv_ls_gd_sgd(gammas, loss_tr_gd, loss_te_gd, loss_tr_sgd, loss_te_sgd)

### LEAST SQUARE RIDGE REGRESSSION

lambdas = np.arange(0.1, 10, 0.1)

print("[Ridge Regression]", end=" ")

t1 = time.time()
best_RR_lambda, loss_tr_rr, loss_te_rr = tune_hyperparam(y, tx, k_fold, seed, hprange=lambdas, method=ridge_regression, args = [], compute_loss=compute_mse)
t2 = time.time()
print("time:", t2 - t1, "best lambda:", best_RR_lambda)
rr_args = [best_RR_lambda]

#plot
cv_ls_rr(lambdas, loss_tr_rr, loss_te_rr)


### LOGISTIC REGRESSION
# logistic == logistic regression
# reg_log == regularized logistic regression
gammas = np.logspace(-12, -4.7, num=15)

print("[Logistic regressions]", end=" ")

t1 = time.time()
log_best_gamma, log_loss_tr, log_loss_te = tune_hyperparam(y, tx, k_fold, seed, hprange=gammas, method=logistic_regression, args=[w_initial, max_iters], compute_loss=neg_log_likelihood)
t2 = time.time()
print("time:", t2 - t1, "Logistic regression best gamma:", log_best_gamma)

# For this one we have to tune both gammas and lambdas.
print("[Regularized logistic regressions]", end=" ")

lambdas = [1,5,10,15,20]
reg_best_loss = np.inf
t1 = time.time()

for lambda_ in lambdas:
	reg_lambda_best_gamma, reg_loss_tr, reg_loss_te = tune_hyperparam(y, tx, k_fold, seed, hprange=gammas, method=reg_logistic_regression, args=[w_initial, lambda_, max_iters], compute_loss=neg_log_likelihood)
	# store gamma and lambda, if the best loss is the best
	if np.argmin(reg_loss_tr) < reg_best_loss:
		reg_best_loss = np.argmin(reg_loss_tr)
		reg_best_lambda = lambda_
		reg_best_gamma = reg_lambda_best_gamma

	# storing losses at lambda == 10 and 20 for plotting the lambda comparison
	if lambda_ == 10:
		loss_l10 = reg_loss_te
	elif lambda_ == 20:
		loss_l20 = reg_loss_te

t2 = time.time()
# find best gamma in reg
print("time:", t2 - t1, "Regularized logistic regression best gamma:", reg_best_gamma, ' ,best lambda:', reg_best_lambda)

cv_reg_log(gammas, log_loss_te, loss_l10, loss_l20)

logistic_args = [w_initial, max_iters, log_best_gamma]
reg_log_args = [w_initial, reg_best_lambda, max_iters, reg_best_gamma]
