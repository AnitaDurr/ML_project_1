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


def least_squares(y, tX):
	None

def ridge_regression(y, tX, lambda_):
	None

####################################################################################################################################
# Logistic regressions (Massimo)

def sigmoid(t):
	"""apply the sigmoid function on t."""
	sig_t = np.exp(t) / (1 + np.exp(t))
	return(sig_t)

def calculate_neg_log_l_loss(y, tX, w):
	"""compute the loss: negative log likelihood."""
	loss = np.mean(np.log(1 + np.exp(tX.dot(w))) - (y * tX.dot(w)))
	return(loss)

def calculate_log_gradient(y, tX, w):
	"""compute the gradient of loss."""
	gradient = tX.T.dot(sigmoid(np.dot(tX, w)) - y)
	return(gradient)

def calculate_hessian(y, tX, w):
	"""return the Hessian of the loss function."""
	S = sigmoid(np.dot(tX, w))
	hessian = tX.T.dot(S*tX)
	return(hessian)

def log_learning_by_gradient_descent(y, tX, w, gamma):
	"""
	Do one step of gradient descent using logistic regression.
	Return the loss and the updated w.
	"""
	loss = calculate_neg_log_l_loss(y, tX, w)
	grad = calculate_log_gradient(y,tX,w)
	# update w
	w = w - (gamma * grad)
	return loss, w

def log_learning_by_newton_method(y, tX, w, gamma):
	"""
	Do one step on Newton's method.
	return the loss and updated w.
	"""
	loss = calculate_neg_log_l_loss(y, tX, w)
	grad = calculate_log_gradient(y, tX, w)
	hess = calculate_hessian(y, tX, w)
	w = w - gamma * (np.linalg.inv(hess).dot(grad))
	return loss, w

def logistic_regression(y, x, max_iter=100, gamma=1., threshold=1e-8, lambda_=0.1, newton_method = False):
	# intiate losses
	losses = []

	# build tX
	tX = np.c_[np.ones((y.shape[0], 1)), x]
	w = np.zeros((tX.shape[1], 1))

	# start the logistic regression
	for iter in range(max_iter):
		# get loss and update w.
		if newton_method == True:
			loss, w = log_learning_by_newton_method(y, tX, w, gamma)
		else:
			loss, w = log_learning_by_gradient_descent(y, tX, w, gamma)

		# log info
		if iter % 1 == 0:
			print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
		# converge criterion
		losses.append(loss)
		if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
			break
	# visualization
	visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_newton_method",True)
	print("loss={l}".format(l=calculate_loss(y, tX, w)))


def reg_logistic_regression(y, tX, lambda_, initial_w, max_iters, gamma):
	None
