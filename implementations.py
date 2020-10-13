#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


### cost functions


def compute_mse(y, tx, w):
	"""
	Compute the mse loss.
	"""
	e = y - tx.dot(w)
	mse = e.dot(e) / (2 * len(e))
	return mse

def compute_rmse(y, tx, w):
	"""
	Compute the rmse loss.
	"""
	return np.sqrt(2 * compute_mse(y, tx, w))

def compute_mae(y, tx, w):
	"""
	Compute the mae loss.
	"""
	e = y - tx.dot(w)
	return np.mean(np.absolute(e))

def compute_gradient(y, tx, w):
	"""
	Gradient of the linear mse function.
	"""
	e = y - tx.dot(w)
	return - tx.T.dot(e) / tx.shape[0]


### ML methods


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
	"""
	Linear regression (mse) using gradient descent.
	"""
	w = initial_w
	for n_iter in range(max_iters):
		# compute gradient
		grad = compute_gradient(y, tx, w)

		# do the gradient step
		w = w - gamma*grad

	# compute final loss
	loss = compute_mse(y, tx, w)

	return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
	"""
	Linear regression (mse) using stochastic gradient descent, with mini-batch size 1.
	"""
	w = initial_w
	for n_iter in range(max_iters):
		# sample a random point and compute its stochastic gradient
		i = np.random.randint(tx.shape[0])
		stoch_grad = compute_gradient(y[i], tx[i], w)

		# do the gradient step
		w = w - gamma*stoch_grad

	# compute final loss
	loss = compute_mse(y, tx, w)

	return w, loss


def least_squares(y, tx):
	"""Least squares solution implemented for ex03 (Massimo)."""
	return(np.linalg.inv(tx.T @ tx) @ tx.T @ y)

def ridge_regression(y, tx, lambda_):
	"""Ridge regression implemented for ex03 (Massimo)."""
	m,n = np.shape(tx)
	I = np.identity((n))
	return(np.linalg.inv(tx.T @ tx + lambda_ * I) @ tx.T @ y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
	None

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	None
