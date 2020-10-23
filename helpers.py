import numpy as np

# 1. Data handling functions
def load_train_data(file):
    X = np.genfromtxt(file, delimiter=",", skip_header=1, usecols=[i for i in range(2,32)])
    y = np.genfromtxt(file, delimiter=",", skip_header=1, usecols=[1], dtype = None, converters={1: lambda x: 0 if b's' in x else 1})
    y = np.expand_dims(y, axis=1)
    return X, y

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

def load_test_data(file):
    X = np.genfromtxt(file, delimiter=",", skip_header=1, usecols=[i for i in range(2,32)])
    return(X)

def sigmoid(t):
    return 1.0 /(1 + np.exp(-t))

def calculate_prediction_log(tx, w):
    predictions = []
    for i in range(tx.shape[0]):
        if sigmoid(tx[i,].dot(w)) > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)

def test_weights(y, x, w, model):
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

    print("  - Test accuracy={a}                       ".format(a=accuracy))
    print("  - Test precision={p}                       ".format(p=precision))
    print("  - Test recall={r}                       ".format(r=recall))
    print("  - Test F1 score={f}                       ".format(f=f1_score))

    return(loss)
