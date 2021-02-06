"""This module contains evaluation metrics for survival analysis
"""

import numpy as np
from scipy.stats import norm

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def precision(cm):
    return cm[1, 1]/(cm[1, 1]+cm[0, 1])


def sensitivity(cm):
    return cm[1, 1]/(cm[1, 1]+cm[1, 0])


def f1score(cm):
    return 2 * precision(cm) * sensitivity(cm) \
           / (precision(cm) + sensitivity(cm))


def lift(cm):
    return precision(cm)/((cm[1, 1]+cm[1, 0])/np.sum(cm))


def accuracy(cm):
    return (cm[0, 0]+cm[1, 1])/np.sum(cm)


def auc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse**(1/2)
    return rmse


def model_performance(y_true: np.array, y_pred: np.array, classification: bool,
                      threshold: float = None):

    if classification:
        if threshold is None:
            raise TypeError("You haven't set a threshold")
        cm = confusion_matrix(
            y_true,
            (y_pred > threshold).astype(int))
        pres = precision(cm)
        sen = sensitivity(cm)
        f_score = f1score(cm)
        li = lift(cm)
        acc = accuracy(cm)
        auc = auc_score(y_true, y_pred)
        # FST rounded the precision of the model performance values
        print(f"Precision is {pres:.4f}\n{'-' * 60}")
        print(f"Sensitivity is {sen:.4f}\n{'-' * 60}")
        print(f"f1score is {f_score:.4f}\n{'-' * 60}")
        print(f"Lift score is {li:.2f}\n{'-' * 60}")
        print(f"Accuracy score is {acc:.2f}\n{'-' * 60}")
        print(f"AUC score is {auc:.2f}\n{'-' * 60}")

    else:
        # Compute performance metrics
        expl_var = explained_variance_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        msle = mean_squared_log_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Print metrics
        print(f"{'Explained variance score:':<30} {expl_var:>10.3f}\n{'-' * 60}")
        print(f"{'Mean absolute error:':<30} {mae:>10.3f}\n{'-' * 60}")
        print(f"{'Mean squared error:':<30} {mse:>10.3f}\n{'-' * 60}")
        print(f"{'Root mean squared error:':<30} {rmse:>10.3f}\n{'-' * 60}")
        print(f"{'Mean squared log error:':<30} {msle:>10.3f}\n{'-' * 60}")
        print(f"{'R2 score:':<30} {r2:>10.3f}\n{'-' * 60}")


def plot_confusion_matrix(y_true, y_pred,
                          threshold, classes,
                          normalize=False,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    FST; I have problems with the centering of
    the numbers in the matrix - screen resolution?
    """
    cm = confusion_matrix(
            y_true,
            (y_pred > threshold).astype(int))
    cmap = plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig, ax = plt.subplots(1, figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax