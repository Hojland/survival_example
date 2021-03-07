
import numpy as np
import pandas as pd
from typing import Any
import os
import re
import pickle
import json
import logging
from functools import partial
from typing import Tuple
import datetime
import torch 
from captum.attr import GradientShap
import shap 

import matplotlib.pyplot as plt
from hyperopt import tpe, Trials, hp, fmin, STATUS_OK
import mlflow
import mlflow.pytorch

from utils import utils, preprocessing_utils, model_utils, wtte_torch
from utils.model_utils import WeibullGRUFitter

import settings

DATA_PATH = "data"
MODELOBJ_PATH = "modelobj"
SEED = 42


class survModel:
    def __init__(
        self,
        feature_names: list,
        params: model_utils.GRUparams=model_utils.GRUparams(),
        local: bool = False,
        seed: int=42,
        scaler: Any=None,
        category_encoder: Any=None,
    ):
        self.params = params
        self.local = local
        self.seed = seed
        self.category_encoder = category_encoder
        self.scaler = scaler
        mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)

    def fit(self, X: np.ndarray, y: np.ndarray, tune_hyperparams: bool=False, experiment_id: str=None):
        input_dim = X.shape[2]
        output_dim = 2

        self.model = WeibullGRUFitter(
            device=settings.DEVICE,
            input_dim=input_dim,
            output_dim=output_dim,
            params=self.params,
            scaler=self.scaler,
            category_encoder=self.category_encoder
        )

        # run the training loop
        if tune_hyperparams:
            tuner = torchAutotuner(
                model=self.model,
                max_evals=10,
                experiment_id=experiment_id,
            )
            tuner.fit(X, y)
            self.params.update(tuner.best_params)
            logging.info("Refitting best parameters")
            self.model.fit(X, y)
        else:
            self.model.fit(X, y)

        if self.local:
            self.local_save(self.model, f"{MODELOBJ_PATH}/model.pickle")
            if tune_hyperparams:
                self.local_save(self.params.to_dict(), f"{MODELOBJ_PATH}/model_config.pickle")

    def eval_loss(self, X, y):
        if self.category_encoder:
            X = self.category_encoder.transform(X)

        eval_loss = self.model.eval(X, y)
        return eval_loss

    def local_save(self, object, file_name):
        if "pickle" in file_name:
            pickle.dump(object, open(file_name, "wb"))
        elif "json" in file_name:
            json.dump(object, open(file_name, "w"))
        else:
            print("format is not supported")

    def shap_values(self, X: np.ndarray, variable: str="alpha"):
        if variable == "alpha":
            var_idx = 0
        elif variable == "beta":
            var_idx = 1
        explainer = GradientShap(self.model)
        shap_values = explainer.attribute(torch.from_numpy(X).float(), torch.from_numpy(X).float(), target=var_idx)
        return shap_values.numpy()

    def log_model(self, **kwargs):
        mlflow.pytorch.log_model( # TODO 
            **kwargs, pytorch_model=self.model, conda_env='utils/resources/conda-env.json'
        )

    def load_model(self, model_name: str, model_stage: str, **kwargs):
        if self.local:
            self.model = pickle.load(open(f"{MODELOBJ_PATH}/model.pickle", "rb"))
        else:
            self.model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_stage}", **kwargs)

    def predict(self, X: np.ndarray):
        return self.model.predict(X).numpy()

    def future_expected_lifetime(self, X: np.ndarray, days_since_event: np.array=None):
        if days_since_event is None:
            days_since_event = np.zeros(len(X))

        out = self.predict(X)
        a, b = out[:, 0], out[:, 1]
        # TODO vectorize this
        return wtte_torch.weibull_expected_future_lifetime(t0=days_since_event, a=a, b=b)

    def future_lifetime_quantiles(self, X: np.ndarray, days_since_event: np.array=None, q: float=0.5):
        if days_since_event is None:
            days_since_event = np.zeros(len(X))
        out = self.predict(X)
        a, b = out[:, 0], out[:, 1]
        return wtte_torch.weibull_future_lifetime_quantiles(q=q, t0=days_since_event, a=a, b=b)

    def future_lifetime_mode(self, X: np.ndarray):
        return NotImplementedError("This method is not yet implemented")


class torchAutotuner(object):
    """Uses hyperopt to perform a Bayesian optimazation of the hyper parameters of our churn model.
    Methods
    --------
    __init__(...) : Construct optimizer
    fit(X,y) : Perform optimization on data
    """
    def __init__(
        self,
        model: WeibullGRUFitter,
        space: dict={},
        max_evals=100,
        n_folds=5,
        n_startup_jobs: int=30,
        rstate=SEED,
        experiment_id: str = None,
    ):
        """Initiates instance of class
        Keyword Arguments:
                    Arguments:
            model {[WeibullGRUFitter]} -- A model that has a reset and params with update function, 
                                          that allows it to save additional values stored on the model object to run.
                                          Furthermore, must have a cv method
            space {[type]} -- The parameter space of the TPE algorithm. If None,
                a predefined space is used.
                Needs to be defined using hyperopt.hp functions.  (default: {None})
            max_evals {int} -- Maximum number of evaluations in TPE algorithm.
                Indicated the total number of optimization steps (including random search).
                (default: {50})
            n_folds {int} -- Number of folds in cv. (default: {5})
            n_startup_jobs {int} -- Number of random search rounds prior to Bayes
                optimization. (default: {30})
            experiment_id {str}: -- Id of the mlflow experiment if used 
        """
        self.experiment_id = experiment_id
        self.model = model
        self.space = {
            "batch_size": 1024,
            "epochs": 300,
            "hidden_dim": hp.choice("hidden_dim", [5, 10, 20, 30]),
            "learn_rate": hp.loguniform("learn_rate", np.log(0.001), np.log(1)),
            "n_layers": hp.choice("n_layers", np.arange(1, 2)),
            "drop_prob": hp.uniform('drop_prob', 0.1, 0.6)
        }
    
        # Setting inputs for TPE algorithm
        if space:
            self.space.update(space)

        self.rstate = rstate
        self.bayes_trials = Trials()
        self.algo = partial(tpe.suggest, n_startup_jobs=n_startup_jobs)
        self.max_evals = max_evals
        self.ITERATION = 0

        # setting inputs for cross validation
        self.n_folds = n_folds

        # Initialize results
        self.best_estimator = None

    def fit(self, X, y):
        """Tune the parameters of predictor.
        Arguments:
            X {np.ndarray} -- Training independent variables
            y {np.ndarray} -- Training dependent variables
        Attributes
        -------
        results : result summary from cv/optimization.
        best_params : best parameters from cv/optimization.
        best_score : best score from cv/optimization.
        """

        # Get objective
        objective_function = lambda params: self._objective_function(
            X, y, 
            params,
            self.n_folds,
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

        self.model.reset()
        self.results = results
        self.best_params = results[0]["params"]
        self.best_score = abs(results[0]["loss"])
        self.model.params.update(self.best_params)

    def _objective_function(
        self, X, y, params, n_folds
    ):
        """Defines the objective function for the bayes optimization. """
        self.ITERATION += 1

        # Perform n_folds cross validation
        mlflow.start_run(experiment_id=self.experiment_id)
        self.model.params.update(params)
        cv_results = self.model.cv(
            X, y,
            n_folds=n_folds,
        )
        # Extract the best score
        loss = np.mean(cv_results['val_loss'])

        # log mlflow metrics
        mlflow.log_metrics({"loss": loss})
        mlflow.log_params({**self.model.params.to_dict()})
        mlflow.end_run()

        return {
            "loss": loss,
            "params": self.model.params.to_dict(),
            "iteration": self.ITERATION,
            "status": STATUS_OK,
        }
