'''A script that compares machine learning methods using for each of them
the best hyperparameters, and select the best model.'''

import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from helpers import *
from tune_hyperparam import *

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def create_boxplot(criterions, methods):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
    count = 0
    xticks = []
    for i, c in enumerate(criterions):

        bp1 = ax.boxplot(c[1], positions = [count+1], widths = [0.7])
        bp2 = ax.boxplot(c[2], positions = [count+2], widths = [0.7])
        bp3 = ax.boxplot(c[3], positions = [count+3], widths = [0.7])
        bp4 = ax.boxplot(c[4], positions = [count+4], widths = [0.7])
        set_box_color(bp1,'#e63946')
        set_box_color(bp2,'#a8dadc')
        set_box_color(bp3,'#457b9d')
        set_box_color(bp4,'#1d3557')

        xticks.append(np.mean([count+1, count+2,count+3,count+4]))

        count += 6

    # temp lines for legend creation
    plt.plot([], c='#e63946', label='Accuracy')
    plt.plot([], c='#a8dadc', label='Precision')
    plt.plot([], c='#457b9d', label='Recall')
    plt.plot([], c='#1d3557', label='F1 score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.ylim((0,1))
    ax.set_xticklabels(methods)
    ax.set_xticks(xticks)
    plt.savefig('Classifiers_comparison.pdf')

def check_convergence(method1, method2, args1, args2):
    weight1, _ = method1(y, tx, *args1)
    weight2, _ = method1(y, tx, *args2)
    return np.linalg.norm(weight2 - weight1)

print("===COMPARE METHODS===")

test_weights_cv = True

seed = 42
k_fold = 4
k_indices = build_k_indices(y, k_fold, seed)

# add arguments for your functions (shoudl actually be variables defined at the enf of the hyperparam tuning)
methods = [least_squares_GD, least_squares_SGD, least_squares, ridge_regression, logistic_regression, reg_logistic_regression]
arguments = [lsGD_args, lsSGD_args, [], rr_args, logistic_args, reg_log_args]

if test_weights_cv:
    print("Following distances between weigths should be small:")
    for i in range(3):
        method1, method2 = methods[i], methods[i+1]
        args1, args2 =  arguments[i], arguments[i+1]
        print(check_convergence(method1, method2, args1, args2))

criterions_df = []
best_accuracy = 0
for i in range(6):
    method = methods[i]
    args = arguments[i]

    accuracy, precision, recall, f1_score = cv_criterions(y, tx, k_indices, method, args)

    criterions_df.append([method.__name__, accuracy, precision, recall, f1_score])

    if np.mean(accuracy) > best_accuracy:
        best_accuracy = np.mean(accuracy)
        best_method = method
        best_args = args

create_boxplot(criterions_df, methods)
