import numpy as np


### cost functions

def compute_mse(y, tX, w):
	"""
	Compute the mse loss.
	"""
	e = y - tX.dot(w)
	mse = e.dot(e) / (2 * len(e))
	return mse

def compute_rmse(y, tX, w):
	"""
	Compute the rmse loss.
	"""
	return np.sqrt(2 * compute_mse(y, tX, w))

def compute_mae(y, tX, w):
	"""
	Compute the mae loss.
	"""
	e = y - tX.dot(w)
	return np.mean(np.absolute(e))

def compute_gradient(y, tX, w):
	"""
	Gradient of the linear mse function.
	"""
	e = y - tX.dot(w)
	return - tX.T.dot(e) / tX.shape[0]


### ML methods

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

# Logisitc regressions:
def sigmoid(t):
    """Applies the sigmoid function"""
    return 1.0 / (1 + np.exp(-t))

def neg_log_likelihood(y, tx, w):
    """compute the loss: negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def logistic_regression(y, x, initial_w, max_iters=1000, gamma=0.01, SGD=False):
    # build tx, initial weights
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    threshold = 1e-8
    losses = []

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

y, X, ids = load_clean_data('data/train.csv')
y = np.expand_dims(y, axis=1)
print(X.shape)
print(y.shape)

def reg_logistic_regression(y, x, initial_w, lambda_, max_iters=1000, gamma=0.01):
    # build tx, initial weights
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1)) + initial_w
    threshold = 1e-8
    losses = []

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
