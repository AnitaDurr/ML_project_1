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
        w = w - gamma * grad
    
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
    None

def ridge_regression(y, tx, lambda_):
    None

####################################################################################################################################
# Logistic regressions (Massimo)

def sigmoid(t):
    """apply the sigmoid function on t."""
    sig_t = np.exp(t) / (1 + np.exp(t))
    return(sig_t)

def calculate_neg_log_l_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    loss = np.mean(np.log(1 + np.exp(tx.dot(w))) - (y * tx.dot(w)))
    return(loss)

def calculate_log_gradient(y, tx, w):
    """compute the gradient of loss."""
    gradient = tx.T.dot(sigmoid(np.dot(tx, w)) - y)
    return(gradient)

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    S = sigmoid(np.dot(tx, w))
    hessian = tx.T.dot(S*tx)
    return(hessian)

def log_learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_neg_log_l_loss(y, tx, w)
    grad = calculate_log_gradient(y,tx,w)
    # update w
    w = w - (gamma * grad)
    return loss, w

def log_learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss = calculate_neg_log_l_loss(y, tx, w)
    grad = calculate_log_gradient(y, tx, w)
    hess = calculate_hessian(y, tx, w)
    w = w - gamma * (np.linalg.inv(hess).dot(grad))
    return loss, w

def logistic_regression(y, x, max_iter=100, gamma=1., threshold=1e-8, lambda_=0.1, newton_method = False):
    # intiate losses
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        if newton_method == True:
            loss, w = log_learning_by_newton_method(y, tx, w, gamma)
        else:
            loss, w = log_learning_by_gradient_descent(y, tx, w, gamma)

        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_newton_method",True)
    print("loss={l}".format(l=calculate_loss(y, tx, w)))


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    None
