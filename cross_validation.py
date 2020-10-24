
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from helpers import *
import time

### functions to get the cross-validation loss

def build_k_indices(y, k_fold, seed):
	"""build k indices for k-fold."""
	np.random.seed(seed)
	num_obs = y.shape[0]
	interval = int(num_obs / k_fold)

	indices = np.random.permutation(num_obs)
	k_indices = [indices[k * interval: (k + 1) * interval]
					for k in range(k_fold)]

	return np.array(k_indices)

def kth_train_test(y, x, k_indices, k):
	"""Returns the k-th subgroup of the data as trainset, and the remaining data as testset"""
	x_te, y_te = x[k_indices[k]], y[k_indices[k]]
	mask = np.ones(x.shape[0], dtype=bool)
	mask[k_indices[k]] = False
	x_tr, y_tr = x[mask], y[mask]

	return x_tr, y_tr, x_te, y_te

def cv_loss(y, x, k_indices, method, args, compute_loss):
	"""
	Compute the cv-losses, using a k-fold cross-validation on the k_indices,
	for the given method called with arguments args and loss computed
	with function compute_loss.
	"""
	loss_tr = []
	loss_te = []

	for k in range(k_fold):
		# build the kth train-test dataset
		x_tr, y_tr, x_te, y_te = kth_train_test(y, x, k_indices, k)

		# learn the weights and compute the train and test loss
		weights, k_loss_tr = method(y_tr, x_tr, *args)
		k_loss_te = compute_loss(y_te, x_te, weights)

		# store the train and test loss
		loss_tr.append(k_loss_tr)
		loss_te.append(k_loss_te)

	# average the k results
	cv_tr = sum(loss_tr) / k_fold
	cv_te = sum(loss_te) / k_fold

	return cv_tr, cv_te


### Functions to tune the hyperparameters and plot the result

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

print('===LOADING DATA===')
y, x, ids = load_clean_data(DATA_TRAIN_PATH)

print('===TUNE HYPERPARAMETERS===')
k_fold = 5
seed = 7

print("[least square GD]", end=" ")

gammas = np.logspace(-10, -1, num=50)
w_initial = np.zeros(x.shape[1])
max_iters = 1000

t1 = time.time()
best_GD_gamma, loss_tr_gd, loss_te_gd = tune_hyperparam(y, x, k_fold, seed, hprange=gammas, method=least_squares_GD, args=[w_initial, max_iters], compute_loss=compute_mse)
t2 = time.time()
cross_validation_visualization(gammas, loss_tr_gd, loss_te_gd, "gamma", "mse loss", "least square GD cross validation")
print("time:", t2 - t1, "best gamma:", best_GD_gamma)


print("[least square SGD]", end=" ")

t1 = time.time()
best_SGD_gamma, loss_tr_sgd, loss_te_sgd = tune_hyperparam(y, x, k_fold, seed, hprange=gammas, method=least_squares_SGD, args=[w_initial, max_iters], compute_loss=compute_mse)
t2 = time.time()
cross_validation_visualization(gammas, loss_tr_sgd, loss_te_sgd, "gamma", "mse loss", "least square SGD cross validation")
print("time:", t2 - t1, "best gamma:", best_SGD_gamma)

cv_gd_sgd(gammas, loss_tr_gd, loss_te_gd, loss_tr_sgd, loss_te_sgd)

