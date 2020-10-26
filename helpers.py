'''
Functions for
_ data processing : loading, cleaning and feature selection
_ loss computations : mse loss, neg_log_likelihood, accuracy, precision, recall f1-score
_ predict labels
'''

import numpy as np
import csv

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

def standardize(x):
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x

def pearson(X, Y):
    return np.cov(X, Y)[0, 1] / (np.std(X) * np.std(Y))

def load_clean_data(file):
    print('Loading data...')
    y, X, ids = load_csv_data(file, sub_sample=True)

    # Data cleaning
    print('Handling missing values...')
    no_val = -999.0
    for j in range(X.shape[1]):
        mean = np.mean(X[:, j] != no_val)
        X[:, j][X[:, j] == no_val] = mean

    # Data standardization
    print('Standardisation...')
    X = standardize(X)

    # Feature selection
    print('Feature selection...')

    # calculates the correlation between the features and returns highly correlated feature couples (pearson > 0.9)
    features_correlation = np.zeros(shape=(X.shape[1],X.shape[1]))
    correlated_feature_couples = []
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            features_correlation[i,j] = pearson(X[:,i],X[:,j])
            if features_correlation[i,j] > 0.9 and i != j and [j,i] not in correlated_feature_couples:
                correlated_feature_couples.append([i,j])

    # iterates over the highly correlated couples and for each finds the feature the less correlated with the label
    features_to_remove = []
    for feature_couple in correlated_feature_couples:
        corr_0 = pearson(y,X[:,feature_couple[0]])
        corr_1 = pearson(y,X[:,feature_couple[1]])
        if corr_0 > corr_1:
            features_to_remove.append(feature_couple[1])
        else:
            features_to_remove.append(feature_couple[0])

    # remove the features in the highly correlated couples that correlated the less with the label
    X = np.delete(X, list(set(features_to_remove)), axis=1)
    # tX = np.c_[np.ones((y.shape[0], 1)), X]

    return y, X, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def compute_mse(y, tX, w):
    """
    Compute the mse loss.
    """
    e = y - tX.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def sigmoid(t):
    """Applies the sigmoid function"""
    return 1.0 / (1 + np.exp(-t))

def neg_log_likelihood(y, tx, w):
    """compute the loss: negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def compute_criterions(y, x, weights, method):
    if method.__name__ in ['logistic_regression', 'reg_logistic_regression']:
        predictions = predict_labels(weights, x, is_LR = True)
    else:
        predictions = predict_labels(weights, x, is_LR = False)

    true_positives = len([i for i in range(len(y)) if (predictions[i] == 1 and y[i] == 1)])
    false_positives = len([i for i in range(len(y)) if (predictions[i] == 1 and y[i] == 0)])
    true_negatives = len([i for i in range(len(y)) if (predictions[i] == 0 and y[i] == 0)])
    false_negatives = len([i for i in range(len(y)) if (predictions[i] == 0 and y[i] == 1)])

    real_pos = len([yi for yi in y if yi == 1])
    real_neg = len([yi for yi in y if yi == 0])
    true_pos_rate = true_positives / real_pos
    true_neg_rate = true_negatives / real_neg
    false_neg_rate = 1 - true_pos_rate
    false_pos_rate = 1 - true_neg_rate

    try:
        accuracy = (true_pos_rate + true_neg_rate) / 2 # balanced accuracy
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = (2 * precision * recall) / (precision + recall)
    except:
        accuracy = np.nan
        precision = np.nan
        recall = np.nan
        f1_score = np.nan

    return accuracy, precision, recall, f1_score

def predict_labels(w, tx, is_LR=False):
    """
    Generates class predictions given weights, and a test data matrix
    """
    if is_LR == True:
        predictions = []
        for i in range(tx.shape[0]):
            pred = sigmoid(tx[i,].dot(w))
            if pred > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return np.array(predictions)

    else:
        y_pred = np.dot(tx, w)
        y_pred[np.where(y_pred <= 0.5)] = 0
        y_pred[np.where(y_pred > 0.5)] = 1
        return y_pred
