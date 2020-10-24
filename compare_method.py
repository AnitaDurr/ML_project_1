'''A script that compares the 6 implemented methods usign for each of them the best hyperparameters.'''
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from helpers import *
from tune_hyperparam import *
import imp

def getVarFromFile(filename):
    f = open(filename)
    global data
    data = imp.load_source('data', '', f)
    f.close()

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

    return [accuracy, precision, recall, f1_score]



y, x, ids = load_clean_data(DATA_TRAIN_PATH)
getVarFromFile(TUNE_PATH)

methods = [least_squares_GD, least_squares_SGD, least_squares, ridge_regression, logistic_regression, reg_logistic_regression]
criterions = np.zeros((5, len(methods)))

print(data.method[0].__name__)

# for i in range(len(methods)):
#     method = methods[i]
#     args = data.method.__name__
#     weights, loss = method(y_tr, x_tr, args)


#     criterions[:i] = [method.__name__] + compute_criterions(y, x, weights, predict, verbose=True)



