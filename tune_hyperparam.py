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
	plt.close()

def cv_log_reg(gamma_range, lambda_range, loss_df, filename, plot_title):
	ax = plt.axes()
	sns.heatmap(loss_df, ax = ax)
	ax.set_title(plot_title)
	ax.set_xticklabels([str(i) for i in gamma_range])
	ax.set_yticklabels([str(i) for i in lambda_range])
	plt.savefig(filename)
	plt.close()

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
# logistic == logistic regression
# reg_log == regularized logistic regression
k_fold = 5
seed = 7
lambdas = np.arange(0, 21, 20)
gammas = np.logspace(-12, -5, num=20)
w_initial = np.zeros(x.shape[1])
max_iters = 100

print("[Logistic regressions]", end=" ")

t1 = time.time()

# For this one we have to tune both gammas and lambdas.
# We use one for both and compute the best gamma after the lambda_ == 0 iteration for simple logistic regression
# Then continue the iterations for penalised logistic regression
for lambda_ in lambdas:
	if lambda_ == 0:
		log_best_gamma, log_loss_tr, log_loss_te = tune_hyperparam(y, x, k_fold, seed, hprange=gammas, method=logistic_regression, args=[w_initial, max_iters], compute_loss=neg_log_likelihood)
		# First iteration: creates numpy arrays that will become data frames for heatmaps
		reg_loss_tr_df = log_loss_tr
		reg_loss_te_df = log_loss_te
		t2 = time.time()
		print("time:", t2 - t1, "Logistic regression best gamma:", log_best_gamma)
		# Start storing best values to keep the best one over lambda iterations
		reg_best_gamma = 0
		reg_best_lambda = 0
		reg_best_loss = np.argmin(log_loss_te)
	else:
		reg_best_gamma, reg_loss_tr, reg_loss_te = tune_hyperparam(y, x, k_fold, seed, hprange=gammas, method=reg_logistic_regression, args=[w_initial, lambda_, max_iters], compute_loss=neg_log_likelihood)
		# Stack this lambda result to the data frames
		reg_loss_tr_df = np.vstack((reg_loss_tr_df,reg_loss_tr))
		reg_loss_te_df = np.vstack((reg_loss_te_df,reg_loss_te))

		if np.argmin(reg_loss_tr) < reg_best_loss:
			reg_best_loss = np.argmin(reg_loss_tr)
			reg_best_lambda = lambda_
			reg_best_gamma = gamma
t3 = time.time()
# find best gamma in reg
print("time:", t3 - t1, "Regularized logistic regression best gamma:", reg_best_gamma)
print(reg_loss_te_df)
print(reg_loss_tr_df)

cv_log_reg(gammas, lambdas, reg_loss_tr_df, 'Reg_log_train.pdf', '-log likelihood for reg. logistic regression: train data')
cv_log_reg(gammas, lambdas, reg_loss_te_df, 'Reg_log_test.pdf', '-log likelihood for reg. logistic regression: test data')

logistic_args = [w_initial, max_iters, log_best_gamma]
reg_log_args = [w_initial, max_iters, reg_best_gamma]



