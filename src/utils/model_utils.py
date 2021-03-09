"""This module contains model utils
"""

from dataclasses import dataclass, asdict
import logging
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any

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

from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import wtte_torch
from utils import utils


@dataclass
class GRUparams:
    """Class for keeping track of an params."""
    epochs: int=400
    learn_rate: float=1e-2
    hidden_dim: int=20
    n_layers: int=2
    batch_size: int=1024
    drop_prob: float=0.2

    def to_dict(self):
        return asdict(self)

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)



class GRUNet(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()

    #def forward(self, X):
    #    out = self.relu(self.fc0(X))
    #    out = self.softplus(self.fc(out))
    #    return out

    def forward(self, X, h: torch.Tensor=None):
        if torch.is_tensor(h) is False:
            h = self.init_hidden(X.shape[0])
        out, h = self.gru(X, h)
        out = self.softplus(self.fc(out[:,-1]))
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden

class WeibullGRUFitter(GRUNet):
    def __init__(self, device, input_dim, output_dim, params: GRUparams=GRUparams(), category_encoder: Any=None, scaler: Any=None):
        super().__init__(device, input_dim, params.hidden_dim, output_dim, params.n_layers, params.drop_prob)
        self.params = params
        self.device = device
        self.criterion = wtte_torch.torchWeibullLoss().loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params.learn_rate)
        self.avg_loss = []
        self.category_encoder = category_encoder
        self.scaler = scaler

    def update_params(self, params: dict):
        self.params.update(params)

    def reset(self):
        """Reset values related to previous model runs"""
        self.avg_loss = []

    def cv(self, X: np.ndarray, y: np.ndarray, n_folds: int=5):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds)
        cv_results = {'val_loss': []}
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            self.fit(X_train, y_train)
            cv_results['val_loss'].append(self.val(X_val, y_val).cpu().detach().numpy())
        return cv_results

    def train_epoch(self, train_loader: DataLoader):
        """Train one epoch of an RNN model"""
        self.train()
        #h = self.init_hidden(self.params.batch_size)
        loss_sum = 0
        counter = 0
        for X, y in train_loader:
            counter += 1
            self.zero_grad()
            #h.detach_()
            out = self.forward(X.to(self.device).float())
            loss = self.criterion(out, y.to(self.device).float())
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
            print(f"loss.item() {loss.item()}")
        self.avg_loss.append(loss_sum/len(train_loader))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train epochs of an RNN model"""
        self.epoch_start_time = utils.time_now()
        train_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.params.batch_size, drop_last=True)

        # Start training loop
        for epoch in range(1, self.params.epochs+1):
            self.train_epoch(train_loader)
            current_time = utils.time_now()
            timed = current_time-self.epoch_start_time
            time_str = utils.strfdelta(timed, "{minutes} minutes & {seconds} seconds")
            if epoch%50 == 0:
                logging.debug(f"Epoch {epoch}/{self.params.epochs} Done, Total AvgNegLogLik: {self.avg_loss[-1]}")
                logging.debug(f"Total Time Elapsed: {time_str} seconds")
        logging.info(f"Total Training Time For Fitting: {time_str} seconds")

    def val(self, X: np.ndarray, y: np.ndarray):
        """ Gets losses from function """
        self.eval()
        with torch.no_grad():
            out = self.predict(X)
            val_loss = self.criterion(out, torch.from_numpy(y).to(self.device).float())
        return val_loss

    def predict(self, X: np.ndarray):
        """ Does prediction """
        self.eval()
        with torch.no_grad():
            h = self.init_hidden(X.shape[0])
            out = self.forward(torch.from_numpy(X).to(self.device).float(), h)
        return out

def target_to_sksurv(duration: np.ndarray, event_indicator: np.ndarray):
    event_indicator = event_indicator.astype(bool)
    dtype = np.dtype([("event", event_indicator.dtype), ("time", duration.dtype)])
    sksurv_array = np.empty(len(event_indicator), dtype=dtype)
    sksurv_array["event"] = event_indicator
    sksurv_array["time"] = duration
    return sksurv_array

def all_evaluation_metrics(
    train_duration: np.ndarray,
    train_event: np.ndarray,
    test_duration: np.ndarray,
    test_event: np.ndarray,
    test_pred: np.ndarray,
):

    survival_train = target_to_sksurv(train_duration, train_event)
    survival_test = target_to_sksurv(test_duration, test_event)
    
    times = np.arange(
        survival_test["time"].min().astype("int"), survival_test["time"].max().astype("int")
    )

    va_auc_arr, mean_auc = cumulative_dynamic_auc(
        survival_train=survival_train,
        survival_test=survival_test,
        estimate=1/test_pred,
        times=times,
        tied_tol=1e-6,
    )
    test_event = test_event.astype(bool)
    c_harrell = concordance_index_censored(
        event_indicator=test_event,
        event_time=test_duration,
        estimate=1/test_pred,
        tied_tol=1e-6,
    )[0]

    c_uno = concordance_index_ipcw(
        survival_train=survival_train,
        survival_test=survival_test,
        estimate=1/test_pred,
        tau=None,
        tied_tol=1e-6,
    )[0]

    res = {
        "c_harrell": c_harrell,
        "c_uno": c_uno,
        "survival_auc_5_10_median": np.median(va_auc_arr[5:10]),
        "survival_auc_10_30_median": np.median(va_auc_arr[10:30]),
        "survival_auc_30_60_median": np.median(va_auc_arr[30:60]),
    }
    return res, va_auc_arr

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