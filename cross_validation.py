
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from helpers import *

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

def cv_loss(y, x, k_indices, k, args, method):
	"""return the cross-validation loss of the specified method."""

	x_tr, y_tr, x_te, y_te = kth_train_test(y, x, k_indices, k)

	if method == 'least_squares_GD':
		weights, loss_tr = least_squares_GD(y_tr, x_tr, *args)
	elif method == 'least_squares_SGD':
		weights, loss_tr = least_squares_SGD(y_tr, x_tr, *args)

	# calculate the mse loss for test data
	loss_te = compute_mse(y_te, x_te, weights)

	return loss_tr, loss_te

### functions to tune and visualize hyperparameters and their corresponding cross-validation losses

def cross_validation_visualization(lambds, mse_tr, mse_te):
	"""
	Visualization the curves of mse_tr and mse_te.
	"""
	plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
	plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
	plt.xlabel("gamma")
	plt.ylabel("mse")
	title = "tuning gamma for least square SGD"
	plt.title(title)
	plt.legend(loc=2)
	plt.grid(True)
	plt.savefig(title)
	plt.show

def tune_hyperparam():
	seed = 7
	k_fold = 4

	max_iters = 50
	w_initial = np.zeros(x.shape[1])
	gammas = np.logspace(-10, -1, num=50)

	# split data in k fold
	k_indices = build_k_indices(y, k_fold, seed)

	# define lists to store the loss of training data and test data
	mse_tr = []
	mse_te = []

	# cross validation
	for ind, gamma in enumerate(gammas):
		loss_tr = []
		loss_te = []
		args = w_initial, max_iters, gamma

		# SGD with a given gammas
		for k in range(k_fold):
			k_loss_tr, k_loss_te = cv_loss(y, x, k_indices, k, args, method='least_squares_SGD')
			loss_tr.append(k_loss_tr)
			loss_te.append(k_loss_te)

		# average the k results
		mse_tr.append(sum(loss_tr) / k_fold)
		mse_te.append(sum(loss_te) / k_fold)

	# keep the gamma giving best mse loss on the test set
	ind_gamma_opt = np.argmin(mse_te)
	best_gamma = gammas[ind_gamma_opt]

	cross_validation_visualization(gammas, mse_tr, mse_te)

	# print(mse_te[ind_gamma_opt])
	return best_gamma



print('===LOADING DATA===')
y, x, ids = load_clean_data(DATA_TRAIN_PATH)
print('===TUNE HYPERPARAMETERS===')
print(tune_hyperparam())
