import numpy as np
import csv
# Script for data loading, data cleaning and feature selection.

# Functions
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
    y, X, ids = load_csv_data(file, sub_sample=False)

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

    #sns.heatmap(features_correlation)
    #plt.title('Pearson correlation for train.csv features')
    #plt.savefig('Features_correlation.pdf')

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
    tX = np.c_[np.ones((y.shape[0], 1)), X]

    return y, tX, ids
