'''Implementation of six machine learning methods.'''

import numpy as np
from helpers import *


def compute_gradient(y, tX, w):
    """
    Gradient of the linear mse function.
    """
    e = y - tX.dot(w)
    return - tX.T.dot(e) / tX.shape[0]


def least_squares_GD(y, tX, initial_w, max_iters, gamma):
	"""
	Linear regression (mse) using gradient descent.
	"""
	w = initial_w
	for n_iter in range(max_iters):
		# compute gradient
		grad = compute_gradient(y, tX, w)

		# do the gradient step
		w = w - gamma * grad

	# compute final loss
	loss = compute_mse(y, tX, w)

	return w, loss

def least_squares_SGD(y, tX, initial_w, max_iters, gamma):
	"""
	Linear regression (mse) using stochastic gradient descent, with mini-batch size 1.
	"""
	w = initial_w
	for n_iter in range(max_iters):
		# sample a random point and compute its stochastic gradient
		i = np.random.randint(tX.shape[0])
		stoch_grad = compute_gradient(y[i], tX[i], w)

		# do the gradient step
		w = w - gamma*stoch_grad

	# compute final loss
	loss = compute_mse(y, tX, w)

	return w, loss

def least_squares(y, tx):
    rank = np.linalg.matrix_rank(tx)
    xtx = np.dot(tx.transpose(), tx)

    #check if a solution exists
    if(rank != xtx.shape[0]):
        print("rank defficient")
        return

    #form and solve the normal equations
    w = np.linalg.solve(xtx, np.dot(tx.transpose(), y))

    #compute the MSE
    loss = compute_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    #form and solve the equations
    a = np.dot(tx.transpose(), tx)
    lambda_p = 2 * len(y) * lambda_
    a = a + np.dot(lambda_p, np.eye(len(y)))
    b = np.dot(tx.transpose(), y)
    w = np.linalg.solve(a,b)

    #compute the RMSE
    loss = compute_rmse(x, y, w)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters=1000, gamma=0.01):
    w = initial_w

    for iter in range(max_iters):
        print('Iteration: {i}'.format(i=iter),end='\r')

        # compute the gradient and update w
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y)
        w = w - gamma * grad

    # Calculate final loss
    pred = sigmoid(tx.dot(w))
    loss = neg_log_likelihood(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, initial_w, lambda_ = 1, max_iters=1000, gamma=0.01):
    w = initial_w

    for iter in range(max_iters):
        print('Iteration: {i}'.format(i=iter),end='\r')

        # compute the gradient, add penalty term and update w
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y) + (2 * lambda_ * w)
        w = w - gamma * grad

    # Calculate final loss, add penalty term
    pred = sigmoid(tx.dot(w))
    loss = neg_log_likelihood(y, tx, w) + (lambda_ * np.squeeze(w.T.dot(w)))
    return w, loss

