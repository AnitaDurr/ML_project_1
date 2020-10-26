# Classification on the CERN Higgs Boson dataset


## What is this project about ?
This project is part of the machine learning course [CS-433](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) given at EPFL during Fall 2020, which we (Bourquin Massimo, Dürr Anita and Krco Natasa.) are following during Fall 2020. With this project, we are introduced to basic concepts of machine learning regarding classification problems.


## What does your script do ?

In this project, we compare different machine learning methods, by comparing their accuracy, precision, recall and f1-score on the train dataset. Once we selected the best method based on this criterions, we predict the labels of the test dataset and write them in a *.csv* file. This file was then uploaded on the [aircrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) corresponding to this project.

As many of our methods have hyperparameters that needs to be tuned, we first use cross-validation method to do this, before comparing the machine learning methods with each other.


### Dataset
We used a [dataset](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf) from the CERN laboratory about the Higgs boson. The two classes indicate whether the observed experience is a result of a Higgs boson or some other process.

The train data contains 250’000 data points: 1 label, 30 features
The test data contains 568'238 data points of the same 30 features.

### Machine Learning methods

Even if all the methods are not really specific to classification problem, we compare the 6 following machine learnings methods, which are implemented in *implementation.py*.

* least squares using a gradient descent: `least_squares_GD`
* least squares using a stochastic gradient descent: `least_squares_SGD`
* least squares using the normal equation: `least_squares`
* ridge regression: `ridge_regression`
* logistic regression using a gradient descent: `logistic_regression`
* regularized logistic regression using a gradient descent: `reg_logistic_regression`

### Step 1 : tune the hyperparameters for each method

Hyperparameters are the gradient step `gamma` for the `least_squares_GD`, `least_squares_SGD`, `logistic_regression` and `reg_logistic_regression` functions and the regularization term `lambda` for the `ridge_regression` and the `reg_logistic_regression` functions.

To tune the hyperparameters, we compute cross-validation losses for different values of hyperparameters, using the `cv_loss` function. We used a `seed = 7` and `k_fold = 4`. For iterative methods we chose to do a maximum of `max_iters = 1000` iterations and to start with all zeros weights `w_initial = np.zeros(x.shape[1])` (you have to start somewhere right, so why not at zero ?).
We also constrained ourselves to test 20 different hyperparameter values.

This tuning of each hyperparameter is done in the *tune_hyperparam.py* script.

### Step 2 : compare the methods

For each method, using the previously tuned hyperparameters, we compute 4 criterions that will allow us to chose the best method : accuracy, precision, recall and f1-score. To compute those criterions we once again use cross-validation with `seed = 42` and `k_fold = 4`.

We boxplot the result in order to choose the best model.

### Step 3 : predict the test labels

This chosen model is finally used to predict the labels of the test dataset. This is simply done by calling the function with the test observations (and of course the tuned hyperparameter(s)).

## How to run the script ?

Download the python scripts in your directory. Train and test datasets should be organized as *.csv* in a subdirectory *data*, i.e. so that their relative paths are *./data/train.csv* and *./data/test.csv*.

Run the *run.py* file. The resulting test labels will appear in a *.csv* file : *./data/predict.csv*.
