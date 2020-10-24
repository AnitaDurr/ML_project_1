'''A script that compares the 6 implemented methods usign for each of them the best hyperparameters.'''
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from data_load_clean_selection import *
from tune_hyperparam import *

def compute_criterions(y, x, weights, predict, verbose=True):
    predictions = predict(weights, x)

    true_positives = len([i for i in range(len(y)) if (predictions[i] == 1 and y[i] == 1)])
    false_positives = len([i for i in range(len(y)) if (predictions[i] == 1 and y[i] == 0)])
    true_negatives = len([i for i in range(len(y)) if (predictions[i] == 0 and y[i] == 0)])
    false_negatives = len([i for i in range(len(y)) if (predictions[i] == 0 and y[i] == 1)])

    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    if verbose:
        print("  - accuracy={a}                       ".format(a=accuracy))
        print("  - precision={p}                       ".format(p=precision))
        print("  - recall={r}                       ".format(r=recall))
        print("  - F1 score={f}                       ".format(f=f1_score))

    return accuracy, precision, recall, f1_score


methods = [least_squares_GD, least_squares_SGD, least_squares, ridge_regression, logistic_regression, reg_logistic_regression]
arguments = [lsGD_args, lsSGD_args, None, None, None, None]
predict_fct = [predict_labels, predict_labels, None, None, None, None]
criterions = np.zeros(len(methods))

for i in range(2):
    method = methods[i]
    args = arguments[i]
    predict = predict_fct[i]
    weights, _ = method(y, x, *args)
    accuracy, precision, recall, f1_score = compute_criterions(y, x, weights, predict)
    criterions[i] = np.array([method.__name__, accuracy, precision, recall, f1_score])



