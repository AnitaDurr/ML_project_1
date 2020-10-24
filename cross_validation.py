'''
Functions to compute cross-validation losses.
'''

import numpy as np
import matplotlib.pyplot as plt
from implementations import *


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
	k_fold = len(k_indices)
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

def cv_criterion(y, x, k_indices, method, args):
	"""
	Compute the cv-losses, using a k-fold cross-validation on the k_indices,
	for the given method called with arguments args and loss computed
	with function compute_loss.
	"""
	k_fold = len(k_indices)
	accuracy_te, precision_te, recall_te, f1_score_te = [], [], [], []

	for k in range(k_fold):
		# build the kth train-test dataset
		x_tr, y_tr, x_te, y_te = kth_train_test(y, x, k_indices, k)

		# learn the weights and compute the test criterions
		weights, _ = method(y_tr, x_tr, *args)
		accuracy, precision, recall, f1_score = compute_criterions(y_te, x_te, weights, predict)

		# store the test criterions
		accuracy_te.append(accuracy)
		precision_te.append(precision)
		recall_te.append(recall)
		f1_score_te.append(f1_score)

	# average the k results
	cv_accuracy = sum(accuracy_te) / k_fold
	cv_precision = sum(precision_te) / k_fold
	cv_recall = sum(recal_te) / k_fold
	cv_f1_score = sum(f1_score_te) / k_fold

	return cv_accuracy, cv_precision, cv_recall, cv_f1_score

