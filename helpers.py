import numpy as np

# Random numbers to seed numpy seeds for cross validation np.random.choice() function, generated with: https://www.calculator.net/random-number-generator.html?slower=1000&supper=9999&ctype=1&s=1764&submit1=Generate
iteration_seeds = [8046,9412,1468,6485,4893,3970,8471,5791,9318,7038]
gammas_to_test = np.logspace(-8,1,num=50)
lambdas_to_test = np.arange(0, 200, 2)

import numpy as np
import csv

# 1. Data handling functions
def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = 0

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def pearson(X, Y):
    return np.cov(X, Y)[0, 1] / (np.std(X) * np.std(Y))

def load_clean_data(file):
    y, X, ids = load_csv_data(file, sub_sample=False)

    ### missing handling data : replace by mean of the observed data
    no_val = -999.0
    for j in range(X.shape[1]):
        mean = np.mean(X[:, j] != no_val)
        X[:, j][X[:, j] == no_val] = mean

    return y, X, ids

def standardize(x):
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def cross_validation_data(y, x, seed, cv_proportion):
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_cv_indices = np.random.choice(num_observations, int(cv_proportion * num_observations),replace=False)

    y_train = y[random_cv_indices]
    x_train = x[random_cv_indices]
    y_test = y[~random_cv_indices]
    x_test = x[~random_cv_indices]
    return y_train, x_train, y_test, x_test

def calculate_predictions(tx, w):
    predictions = []
    for i in range(tx.shape[0]):
        if sigmoid(tx[i,].dot(w)) > 0.5:
            predictions.append(1)
        else:
            predictions.append(-1)
    return np.array(predictions)

def test_weights(y, x, w):
    tx = np.c_[np.ones((y.shape[0], 1)), x]

    loss = calculate_loss(y, tx, w)
    print("  - Test loss={l}                           ".format(l=loss))

    if model == 'logistic':
        predictions = calculate_prediction_log(tx, w)
    # to create prediction functions for other methods

    # true_positives = np.sum([1 if predictions[i] == 1 and y[i] == 1 else 0 for i in range(len(y))])
    # false_positives = np.sum([1 if predictions[i] == 1 and y[i] == 0 else 0 for i in range(len(y))])
    # true_negatives = np.sum([1 if predictions[i] == 0 and y[i] == 0 else 0 for i in range(len(y))])
    # false_negatives = np.sum([1 if predictions[i] == 0 and y[i] == 1 else 0 for i in range(len(y))])

    # true_positives = len([predictions[i] == 1 and y[i] == 1 for i in range(len(y))])
    # false_positives = len([predictions[i] == 1 and y[i] == 0 for i in range(len(y))])
    # true_negatives = len([predictions[i] == 0 and y[i] == 0 for i in range(len(y))])
    # false_negatives = len([predictions[i] == 0 and y[i] == 1 for i in range(len(y))])

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0


    for i in range(len(y)):
        if predictions[i] == 1 and y[i] == 1:
            true_positives += 1
        elif predictions[i] == 1 and y[i] == 0:
            false_positives += 1
        elif predictions[i] == 0 and y[i] == 0:
            true_negatives += 1
        else:
            false_negatives += 1


    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print("  - accuracy={a}                       ".format(a=accuracy))
    print("  - precision={p}                       ".format(p=precision))
    print("  - recall={r}                       ".format(r=recall))
    print("  - F1 score={f}                       ".format(f=f1_score))
    return([loss, accuracy, precision, recall, f1_score])
