
import numpy as np
import pandas as pd
from captum.attr import GradientShap # could also include Shapley Value Sampling
import os
import re
import pickle
import json
import logging
from functools import partial
from typing import Tuple
import datetime

import matplotlib.pyplot as plt
from hyperopt import tpe, Trials, hp, fmin, STATUS_OK
import mlflow

from utils import utils, preprocessing_utils, model_utils
from utils.model_utils import WeibullGRUFitter

import settings

DATA_PATH = "data"
MODELOBJ_PATH = "modelobj"
SEED = 42


class survModel:
    def __init__(
        self,
        params: model_utils.GRUparams=model_utils.GRUparams(),
        local: bool = False,
        seed: int=42,
    ):
        self.params = params
        self.local = local
        self.seed = seed
        mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)

    def fit(self, X: np.ndarray, y: np.ndarray, tune_hyperparams: bool=False, experiment_id: str=None):
        input_dim = X.shape[2]
        output_dim = 2

        self.model = WeibullGRUFitter(
            device=settings.DEVICE,
            input_dim=input_dim,
            output_dim=output_dim,
            params=self.params,
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

    def make_SHAP(self, X: pd.DataFrame, y: pd.DataFrame=None):
        def get_baselines(X):
            out = self.model.predict(X)
            return out.mean(dim=0).repeat(out.shape[0], 1)
        baselines = get_baselines(X)
        explainer = GradientShap(self.model.model)
        attribution = explainer.attribute(torch.Tensor(X), baselines=baselines, target=1)


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
