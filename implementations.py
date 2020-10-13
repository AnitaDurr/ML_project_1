import numpy as np


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

def mse_gradient(y, tx, w):
	"""
	Gradient of the linear mse function.
	"""
	e = y - tx.dot(w)
	return - tx.T.dot(e) / y.shape[0]


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
	"""
	Linear regression using gradient descent.
	"""
	return w, loss

def least_squares SGD(y, tx, initial w, max iters, gamma):
	"""
	Linear regression using stochastic gradient descent, with mini-batch size 1.
	"""
	None

def least_squares(y, tx):
    """Least squares solution implemented for ex03 (Massimo)."""
    return(np.linalg.inv(tx.T @ tx) @ tx.T @ y)

def ridge_regression(y, tx, lambda_):
    """Ridge regression implemented for ex03 (Massimo)."""
    m,n = np.shape(tx)
    I = np.identity((n))
    return(np.linalg.inv(tx.T @ tx + lambda_ * I) @ tx.T @ y)

def logistic_regression(y, tx, initial w, max iters, gamma):
	None

def reg_logistic_regression(y, tx, lambda_, initial w, max iters, gamma):
	None


