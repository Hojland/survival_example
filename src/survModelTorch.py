
import numpy as np
import pandas as pd
import shap
import os
import re
import pickle
import json
import logging
from functools import partial
from typing import Tuple
import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from hyperopt import tpe, Trials, hp, fmin, STATUS_OK
import mlflow

from utils import utils, preprocessing_utils, model_utils
from utils import wtte_torch
from torch.utils.data import TensorDataset, DataLoader
from survModelTorch import GRUNet

import settings

DATA_PATH = "data"
MODELOBJ_PATH = "modelobj"
SEED = 42


class survModel:
    def __init__(
        self,
        params: model_utils.GRUparams,
        local: bool = False,
        seed: int=42,
        batch_size: int=1024,
    ):
        self.params = params
        self.local = local
        self.seed = seed
        self.batch_size = batch_size
        mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)

    def fit(self, X: np.ndarray, y: np.ndarray, tune_hyperparams: bool=False, experiment_id: str=None):
        def train_epoch(self, train_loader: DataLoader):
            """Train one epoch of an RNN model"""
            self.start_time = utils.time_now()
            h = self.model.init_hidden(self.params.batch_size)
            avg_loss = 0
            counter = 0
            for x, y in train_loader:
                counter += 1
                self.model.zero_grad()
                h.detach_()

                out, h = self.model(x.to(settings.DEVICE).float(), h)
                loss = self.criterion(out, y.to(settings.DEVICE).float())
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
            return avg_loss #also return out, h?

        def train_epochs(self, train_loader: DataLoader):
            """Train epochs of an RNN model"""
            epoch_times = []
            # Start training loop
            for epoch in range(1, self.params.epochs+1):
                current_time = utils.time_now()
                avg_loss = train_epoch(self, train_loader)
                logging.info(f"Epoch {epoch}/{self.params.EPOCHS} Done, Total AvgNegLogLik: {avg_loss/len(train_loader)}")
                logging.info(f"Total Time Elapsed: {str(current_time-self.start_time)} seconds")
                epoch_times.append(current_time-self.start_time)
            logging.info(f"Total Training Time: {str(sum(epoch_times, datetime.timedelta()))} seconds")

        input_dim = X.shape[2]
        output_dim = 2
        train_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.params.batch_size, drop_last=True)
        
        # run the training loop
        if tune_hyperparams:
            raise NotImplementedError()
        else:
            # TODO these won't show if I do tune hyperparams, since we would need to init model from within. Maybe output these
            # in the tuning class fitting too. Or just pass them along not as model objects. and train_epoch and train_epochs should maybe be moved. Maybe to model_utils - 
            # Could be in there as a WeibullGRUFitter class, which could be used also in the hyperparameter updater
            self.model = GRUNet(settings.DEVICE, input_dim, self.params.hidden_dim, output_dim, self.params.n_layers).to(settings.DEVICE)
            self.criterion = wtte_torch.torchWeibullLoss().loss
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learn_rate)
            train_epochs(self, train_loader)

        if self.local:
            self.local_save(self.model, f"{MODELOBJ_PATH}/model.pickle")
            if tune_params:
                self.local_save(self.params, f"{MODELOBJ_PATH}/model_config.pickle")

    #def eval_loss(self, X, y): # TODO replace this with just getting negloklik from wtte_torch
    #    if self.category_encoder:
    #        X = self.category_encoder.transform(X)
#
    #    dtrain = xgb.DMatrix(X, y)
    #    eval_res = self.model.eval(dtrain)
    #    eval_name = re.search("(?<=\\t)([a-z-]*)(?<!:)", eval_res)
    #    eval_loss = float(re.search("(?<=:)([\d\.]*)", eval_res)[0])
    #    return eval_loss

    def local_save(self, object, file_name):
        if "pickle" in file_name:
            pickle.dump(object, open(file_name, "wb"))
        elif "json" in file_name:
            json.dump(object, open(file_name, "w"))
        else:
            print("format is not supported")

    def make_SHAP(self, X: pd.DataFrame, y: pd.DataFrame=None):
        raise NotImplementedError()
        #explainer = shap.TreeExplainer(
        #    self.model,
        #    data=X,
        #    model_output=model_output,
        #    feature_dependence="independent",
        #) KERNELEXPLAINER

        #if self.local:
        #    self.local_save(explainer, f"{MODELOBJ_PATH}/explainer.pickle")
        #    shap_values = explainer.shap_values(X, approximate=True)
        #    self.local_save(shap_values, f"{MODELOBJ_PATH}/shap_values.pickle")

    def log_model(self, **kwargs):
        if self.category_encoder:
            self._save_encoder(path="encoder.pkl")
            mlflow.pyfunc.log_model(
                **kwargs, python_model=self, artifacts={"encoder": "encoder.pkl"}
            )
        else:
            mlflow.pyfunc.log_model(python_model=self, **kwargs)

    def _save_encoder(self, path="encoder.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.category_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, model_name: str, model_stage: str, **kwargs):
        if self.local:
            self.model = pickle.load(open(f"{MODELOBJ_PATH}/model.pickle", "rb"))
        else:
            pyfunc = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}", **kwargs)
            artifacts = pyfunc._model_impl.context._artifacts
            self.model = pyfunc._model_impl.python_model.booster

            if "encoder" in artifacts.keys():
                with open(artifacts["encoder"], "rb") as f:
                    self.category_encoder = pickle.load(f)

    def print_performance(self, X: pd.DataFrame, y: pd.DataFrame, threshold: float=None):
        raise NotImplementedError()
        # performance metrics

    def plot_predictions(self, X: pd.DataFrame, y: pd.DataFrame):
        raise NotImplementedError()
        ## metrics
#
        #plt.scatter(y, y_pred)
#
        #plt.gca().set_aspect("equal", adjustable="box")
#
        #plt.show()

    def plot_confusion_matrix(self, X: pd.DataFrame, y: pd.DataFrame, threshold: float):
        raise NotImplementedError()
        # plot at specific times difference

        #model_utils.\
        #    plot_confusion_matrix(y, y_pred, threshold,
        #                          [0, 1],
        #                          title='Confusion matrix')
        #return plt.show()

    def SHAP_summary(self, X: pd.DataFrame, max_display: int = None):
        raise NotImplementedError()
        # how should we reference and safe our stuff?
        #return shap.summary_plot(self.shap_values,
        #                         features=X,
        #                         max_display=max_display)


class GRUNet(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        #self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.softplus(self.fc(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


class torchAutotuner(object):
    """Uses hyperopt to perform a Bayesian optimazation of the hyper parameters of our churn model.
    Methods
    --------
    __init__(...) : Construct optimizer
    fit(X,y) : Perform optimization on data
    """

    def __init__(
        self,
        objective,
        eval_metric,
        space: dict={},
        max_evals=100,
        nfold=5,
        n_startup_jobs=20,
        csv_file_name=None,
        rstate=SEED,
        num_boost_round=1000,
        early_stopping_rounds=50,
        experiment_id: str = None,
    ):
        """Initiates instance of class
        Keyword Arguments:
                    Arguments:
            objective {string} -- A xgb or lgb approved objective name. See lgb/xgb documentation.
            eval_metric {string} -- A xgb or lgb approved eval_metric. See lgb/xgb documentation.
            space {[type]} -- The parameter space of the TPE algorithm. If None,
                a predefined space is used.
                Needs to be defined using hyperopt.hp functions.  (default: {None})
            max_evals {int} -- Maximum number of evaluations in TPE algorithm.
                Indicated the total number of optimization steps (including random search).
                (default: {50})
            nfold {int} -- Number of folds in cv. (default: {5})
            n_startup_jobs {int} -- Number of random search rounds prior to Bayes
                optimization. (default: {30})
            csv_file_name {[type]} -- Filename of .csv-file the result of each optimization
                round is printed to. If None, no output is produced (default: {None})
            rstate {[type]} -- Random number seed (default: {None})
            num_boost_round {int} -- Maximum number of boosted trees to fit. (default: {5000})
            early_stopping_rounds {int} -- Activates early stopping. The model will train
                until the validation score stops improving. Validation score needs to improve
                at least every early_stopping_rounds round(s)
                to continue training. (default: {500})
            eval_metric {[type]} -- Evaluation metrics for optimization. If None,
                than default metric for objective is used. (default: {None})
            experiment_id {str}: -- Id of the mlflow experiment if used 
        """
        self.experiment_id = experiment_id
        self.space = {
            "tree_method": "hist",
            "subsample": hp.uniform("subsample", 0.6, 1),
            "max_depth": hp.choice("max_depth", np.arange(2, 15, 1)),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(1)),
            "lambda": hp.loguniform("lambda", np.log(0.001), np.log(100)),
            "alpha": hp.loguniform("alpha", np.log(0.001), np.log(100)),
            "min_child_weight": hp.loguniform(
                "min_child_weight", np.log(0.001), np.log(100)
            ),
        }
        
        # Setting inputs for TPE algorithm
        self.space.update({"obj": objective, 
                           "feval": eval_metric})
        if space:
            self.space.update(space)

        self.rstate = rstate
        self.bayes_trials = Trials()
        self.algo = partial(tpe.suggest, n_startup_jobs=n_startup_jobs)
        self.max_evals = max_evals
        self.csv_file_name = csv_file_name
        self.ITERATION = 0

        # setting inputs for cross validation
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.nfold = nfold
        self.eval_metric = eval_metric

        # Initialize results
        self.best_estimator = None

    def fit(self, dtrain):
        """Tune the parameters of predictor.
        Arguments:
            dtrain {cudf.DataFrame} -- Training data
        Attributes
        -------
        results : result summary from cv/optimization.
        best_params : best parameters from cv/optimization.
        best_score : best score from cv/optimization.
        """

        # Get objective
        objective_function = lambda params: self._objective_function(
            params,
            dtrain,
            self.num_boost_round,
            self.early_stopping_rounds,
            self.nfold,
            self.eval_metric,
        )
        # Optimize model
        fmin(
            fn=objective_function,
            space=self.space,
            algo=self.algo,
            max_evals=self.max_evals,
            trials=self.bayes_trials,
            rstate=np.random.RandomState(self.rstate),
        )
        # Results ordered in descending order of performance
        results = sorted(self.bayes_trials.results, key=lambda x: x["loss"])

        self.results = results
        self.best_params = results[0]["params"]
        self.best_score = abs(results[0]["loss"])
        self.num_boost_round = results[0]["num_boost_round"]

    def _objective_function(
        self, params, train_set, num_boost_round, early_stopping_rounds, nfold, eval_metric
    ):
        """Defines the objective function for the bayes optimization. """
        self.ITERATION += 1

        start_time = utils.time_now()
        # Perform n_folds cross validation
        #mlflow.start_run(experiment_id=self.experiment_id)
        cv_results = xgb.cv(
            params,
            dtrain=train_set,
            num_boost_round=num_boost_round,
            nfold=nfold,
            early_stopping_rounds=early_stopping_rounds,
            feval=eval_metric,
            seed=SEED,
            shuffle=True,
        )
        # Extract the best score
        key = list(cv_results.keys())[0]
        loss = np.min(cv_results[key])
        num_boost_round = int(np.argmin(cv_results[key]) + 1)

        # log mlflow metrics
        #mlflow.log_metrics({"loss": loss})
        #mlflow.log_params({**params, "num_boost_round": num_boost_round})
        #mlflow.end_run()

        return {
            "loss": loss,
            "params": params,
            "iteration": self.ITERATION,
            "num_boost_round": num_boost_round,
            "status": STATUS_OK,
        }
