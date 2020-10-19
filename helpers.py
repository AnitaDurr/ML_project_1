import numpy as np

def load_train_data(file):
    X = np.genfromtxt(file, delimiter=",", skip_header=1, usecols=[i for i in range(2,32)])
    y = np.genfromtxt(file, delimiter=",", skip_header=1, usecols=[1], dtype = None, converters={1: lambda x: 0 if b's' in x else 1})
    return(X, y)

def load_test_data(file):
    X = np.genfromtxt(file, delimiter=",", skip_header=1, usecols=[i for i in range(2,32)])
    return(X)
