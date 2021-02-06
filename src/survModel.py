import pickle
import json
import matplotlib.pyplot as plt
import logging
from hyperopt import tpe, Trials, hp, fmin, STATUS_OK
from functools import partial
from typing import Tuple
import mlflow
import xgboost as xgb
import numpy as np
import pandas as pd
import shap
import os
import re

from utils import utils, preprocessing_utils, evaluation_metrics
from utils import wtte
import settings

DATA_PATH = "data"
MODELOBJ_PATH = "modelobj"
SEED = 42


class survModel:
    def __init__(
        self,
        local: bool = False,
        params: dict = {},
        train_params: dict = {},
        n_jobs: str=1,
        model_framework: str="xgboost",
        seed: int=42,
        modeltype: str="survival",
        tune_params: bool=True
    ):
        self.model_framework = model_framework
        self.modeltype = modeltype
        self.local = local
        self.n_jobs = n_jobs
        self.seed = seed
        self.params = params
        self.history = None
        self.train_params = train_params
        mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)

    #def convert_label_format(self, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    #    """Convert from censored start stop format to label_lower_bound and label_upper_bound
    #    as required by xgboost"""
    #    label_lower_bound = y['start'].values
    #    label_upper_bound = np.where(y['censored']==1, np.inf, y['stop'])
    #    return label_lower_bound, label_upper_bound


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, tune_params: bool=False, experiment_id: str=None):

        if self.modeltype == 'survival':
            objective = wtte.weibull_timevaryingcov_continuous_obj
            eval_metric = wtte.weibull_timevaryingcov_continuous_loglik_m
            self.params.update({'disable_default_eval_metric': 1})
            y['censored'] = y['censored'].astype('int')
            
            #label_lower_bound, label_upper_bound = self.convert_label_format(y)

            dtrain = xgb.DMatrix(X)
            dtrain.index = X.index
            dtrain.censored = y['censored'] * 1
            dtrain.start = y['start']
            dtrain.stop = y['stop']

            #dtrain.set_float_info('label_lower_bound', label_lower_bound)
            #dtrain.set_float_info('label_upper_bound', label_upper_bound)
        else:
            dtrain = xgb.DMatrix(X, y)


        if self.model_framework == "xgboost":
            if tune_params:
                # objective should be own function
                tuner = xgbAutotuner(
                    objective=objective,
                    eval_metric=eval_metric,
                    space=self.params,
                    num_boost_round=100,
                    early_stopping_rounds=100,
                    experiment_id=experiment_id,
                )
                tuner.fit(dtrain)
                self.train_params["num_boost_round"] = tuner.num_boost_round
                self.params.update(**tuner.best_params)
            else:
                evals = [(dtrain, "train")]
                output = xgb.train(
                    {**self.params},
                    obj=objective,
                    feval=eval_metric,
                    dtrain=dtrain,
                    **self.train_params,
                    evals=evals,
                    evals_result=self.history,
                )

                self.model = output["booster"]
                self.history = output["history"]

        elif self.model_framework == "catboost":
            raise NotImplementedError()
        elif self.model_framework == "lightgbm":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        if self.local:
            self.local_save(self.model, f"{MODELOBJ_PATH}/model.pickle")
            if tune_params:
                self.local_save(self.params, f"{MODELOBJ_PATH}/model_config.pickle")

    def eval_loss(self, X, y):
        if self.category_encoder:
            X = self.category_encoder.transform(X)

        dtrain = xgb.DMatrix(X, y)
        eval_res = self.model.eval(dtrain)
        eval_name = re.search("(?<=\\t)([a-z-]*)(?<!:)", eval_res)
        eval_loss = float(re.search("(?<=:)([\d\.]*)", eval_res)[0])
        return eval_loss

    def local_save(self, object, file_name):
        if "pickle" in file_name:
            pickle.dump(object, open(file_name, "wb"))
        elif "json" in file_name:
            json.dump(object, open(file_name, "w"))
        else:
            print("format is not supported")

    def make_SHAP(self, X: pd.DataFrame, y: pd.DataFrame=None):
        if self.modeltype == 'classification':
            model_output = "probability"
        elif self.modeltype == 'regression':
            model_output = "margin"
        elif self.modeltype == 'survival':
            model_output = "margin"

        if self.modeltype == 'catboost':
            from catboost import Pool
            cat_features = preprocessing_utils.determine_cat(X)
            shap_values = self.model.get_feature_importance(
                data=Pool(X, y, cat_features=cat_features),
                fstr_type='ShapValues', verbose=False)
            self.shap_values = shap_values[:, :-1]
            self.local_save(self.shap_values, f"{MODELOBJ_PATH}/shap_values.pickle")
            return
        else:
            explainer = shap.TreeExplainer(
                self.model,
                data=X,
                model_output=model_output,
                feature_dependence="independent",
            )

        if self.local:
            self.local_save(explainer, f"{MODELOBJ_PATH}/explainer.pickle")
            shap_values = explainer.shap_values(X, approximate=True)
            self.local_save(shap_values, f"{MODELOBJ_PATH}/shap_values.pickle")

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
        if self.modeltype == 'classification':
            y_pred = self.model.predict_proba(X)[:, 1]
        elif self.modeltype == 'regression':
            y_pred = self.model.predict(X)
        elif self.modeltype == 'survival':
            y_pred = self.model.predict(X) 
            # TODO NEEDS MORE

        #evaluation_metrics.model_performance(y, y_pred,
        #                              modeltype=self.modeltype,
        #                              threshold=threshold)

    def plot_predictions(self, X: pd.DataFrame, y: pd.DataFrame):
        if self.modeltype == 'classification':
            y_pred = self.model.predict_proba(X)[:, 1]
        else:
            y_pred = self.model.predict(X)

        plt.scatter(y, y_pred)

        plt.gca().set_aspect("equal", adjustable="box")

        plt.show()

    def plot_confusion_matrix(self, X: pd.DataFrame, y: pd.DataFrame, threshold: float):
        if self.modeltype == 'classification':
            y_pred = self.model.predict_proba(X)[:, 1]
        else:
            y_pred = self.model.predict(X)

        evaluation_metrics.\
            plot_confusion_matrix(y, y_pred, threshold,
                                  [0, 1],
                                  title='Confusion matrix')
        return plt.show()

    def SHAP_summary(self, X: pd.DataFrame, max_display: int = None):
        return shap.summary_plot(self.shap_values,
                                 features=X,
                                 max_display=max_display)


class xgbAutotuner(object):
    """Uses hyperopt to perform a Bayesian optimazation of the hyper parameters of an XGBoost_gpu.
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
