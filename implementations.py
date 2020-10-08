import numpy as np

def least_sqaures_GD(y, tx, initial_w, max_iters, gamma):
	None

def least_squares SGD(y, tx, initial w, max iters, gamma):
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


