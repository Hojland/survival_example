%load_ext autoreload
%autoreload 2
import pandas as pd
import shap
import torch
import boto3
import numpy as np
import mlflow
import re
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import sqlalchemy
import logging
from sklearn.preprocessing import Normalizer

from utils import sql_utils, utils, plot_utils, model_utils, preprocessing_utils
from survModelTorch import survModel
import settings


SQL_TABLE_NAME = "model_values"
SQL_SCHEMA = "models"
MODEL_NAME = "mart-surv"
DATASET_NAME = "../data/telco_cust_surv_churn.csv"

def get_data():
    df = pd.read_csv(DATASET_NAME)
    df['not_censored'] = 1 - df['censored']
    df['t'] = df['stop'] - df['start']
    df = df.drop(['start', 'stop', 'censored'], axis=1)
    return df

def main():
    logger = utils.get_logger("printyboi.log")

    db_engine = sqlalchemy.create_engine('sqlite:///data/surv.db')
    mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)
    df = get_data()

    df = df.set_index(['customerID'])

    X, y = preprocessing_utils.split_X_y(df, y_variables=['not_censored', 't'])
    X = pd.get_dummies(X, drop_first=True)
    shap_X = X.copy()
    shap_X['t'] = y['t']
    feature_names = list(X)
    shap_X_feature_names = list(shap_X)    

    X_train, X_test, y_train, y_test = preprocessing_utils.train_test_split(X, y, settings.TEST_SIZE, settings.SEED)

    non_bin_cols = utils.find_binary(X, get_difference=True)
    scaler = Normalizer().fit(X_train[non_bin_cols])
    scaler.non_bin_cols = non_bin_cols
    X_train[non_bin_cols] = scaler.transform(X_train[non_bin_cols])
    X_test[non_bin_cols] = scaler.transform(X_test[non_bin_cols])

    X_train, X_test, y_train, y_test = preprocessing_utils.to_cubes(X_train, X_test, y_train, y_test, max_seq_len=2)

    logger.info("fitting the model")
    experiment_id = mlflow.create_experiment(
        f"mart-surv-39"
    )
    params = model_utils.GRUparams(epochs=300, learn_rate=0.042678505501429896, hidden_dim=5, n_layers=1, batch_size=1024, drop_prob=0.15572706273371986)
    surv = survModel(params=params, feature_names=feature_names, scaler=scaler)
    surv.fit(X=X_train, y=y_train, tune_hyperparams=False, experiment_id=experiment_id)

    # TODO change shap to a dataframe and add it so db
    logger.info("refitting the model to best params")
    with mlflow.start_run(experiment_id=experiment_id):
        os.makedirs("tmp_artifacts", exist_ok=True)
        mlflow.set_tags({"lifecycle": "FINAL_RUN"})
        mlflow.log_params(surv.params.to_dict())
        alpha_shap, beta_shap = surv.shap_values(X_test, variable='alpha'), surv.shap_values(X_test, variable='beta')
        np.save("tmp_artifacts/alpha_shap_values", alpha_shap)
        np.save("tmp_artifacts/beta_shap_values", beta_shap)
        mlflow.log_artifact(local_path="tmp_artifacts/alpha_shap_values.npy")
        mlflow.log_artifact(local_path="tmp_artifacts/beta_shap_values.npy")

        # plotting summary for the last event
        shap.summary_plot(alpha_shap[:,-1,:], features=X_test[:,-1,:], feature_names=feature_names, max_display=10, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig("tmp_artifacts/alpha_shap_fig.svg")
        mlflow.log_artifact(local_path="tmp_artifacts/alpha_shap_fig.svg")

        # plotting summary for the last event
        shap.summary_plot(beta_shap[:,-1,:], features=X_test[:,-1,:], feature_names=feature_names, max_display=10)
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig("tmp_artifacts/beta_shap_fig.svg")
        mlflow.log_artifact(local_path="tmp_artifacts/beta_shap_fig.svg")

        tmp = alpha_shap.transpose(1,0,2).reshape(alpha_shap.shape[0] * alpha_shap.shape[1], -1, order='F')
        tmp = pd.DataFrame(tmp, columns=feature_names)
        tmp['t'] = y_test[:,:,1].transpose(1,0,2)
        #shap.initjs()
        #shap.force_plot(explainer.expected_value[0], alpha_shap[:,-1,:], X[:,-1,:], feature_names=feature_names)
        #shap.force_plot(explainer.expected_value[0], alpha_shap[0,-1,:], X[0,-1,:], feature_names=feature_names)

        y_pred = surv.future_lifetime_quantiles(X_test)
        res, survival_auc_arr = model_utils.all_evaluation_metrics(
            y_train[:, -1, 1],
            y_train[:, -1, 0],
            y_test[:, -1, 1],
            y_test[:, -1, 0],
            y_pred,
        )
        survival_auc_arr_plot_df = pd.DataFrame(survival_auc_arr)
        survival_auc_arr_plot_df = survival_auc_arr_plot_df.reset_index().rename(
            {"index": "days", 0: "survival_auc"}, axis=1
        )
        cm = LinearSegmentedColormap.from_list("nuuday", plot_utils.nuuday_palette()[0:5])
        font = fm.FontProperties(fname="utils/resources/Nuu-Light.ttf")
        ax = survival_auc_arr_plot_df.plot(x="days", y="survival_auc", colormap=cm)
        ax = plot_utils.set_ax_style(ax, font)
        fig = ax.get_figure()
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig("tmp_artifacts/survival_auc.svg", transparent=True)
        mlflow.log_artifact(local_path="tmp_artifacts/survival_auc.svg")

        mlflow.log_metrics(res)
        surv.log_model(
            artifact_path=f"obscure/{MODEL_NAME}",
            registered_model_name=MODEL_NAME,
        )
        os.system("rm -rf tmp_artifacts")

    res["eval_loss"] = surv.eval_loss(X_test, y_test)

    res.update(surv.params.to_dict())
    res.update(
        {
            "time": utils.time_now().strftime("%Y-%m-%d"),
            "model_name": MODEL_NAME,
        }
    )
    res = pd.DataFrame(res, index=[0])
    dtype_trans_dct = sql_utils.get_dtype_trans_notalchemy(res)
    dtype_trans_dct["time"] = "DATE"
    sql_utils.create_table(
        db_engine, f"{SQL_SCHEMA}.{SQL_TABLE_NAME}", col_datatype_dct=dtype_trans_dct,
    )
    sql_utils.df_to_sql(db_engine, res, f"{SQL_SCHEMA}.{SQL_TABLE_NAME}")


if __name__ == '__main__':
    main()

# TODO
# make readme
# make dashboard in dash