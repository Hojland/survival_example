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
import category_encoders as ce

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
    df[df['t']==0] = 1

    X, y = preprocessing_utils.split_X_y(df, y_variables=['not_censored', 't'])
    
    # target encoding
    y_pred = preprocessing_utils.dur_model_target(y['t'], y['not_censored'])
    cat_cols = list(X.select_dtypes(["category", "object"]).columns)
    bin_cols = utils.find_binary(X[set(X) - set(cat_cols)], get_difference=False)
    encode_cols = bin_cols + cat_cols
    category_encoder = ce.TargetEncoder(cols=encode_cols)
    category_encoder = category_encoder.fit(X, y_pred)
    X = category_encoder.transform(X)

    # Scale variables
    index = X.index
    feature_names = list(X)
    scaler = Normalizer().fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=feature_names, index=index)

    X_train, X_test, y_train, y_test = preprocessing_utils.train_test_split(X, y, settings.TEST_SIZE, settings.SEED)
    X_train, X_test, y_train, y_test = preprocessing_utils.to_cubes(X_train, X_test, y_train, y_test, max_seq_len=2)

    # testing out just fully connected, remove in GRU!
    y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1], y_train.shape[2])
    X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2])
    y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2])
    X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])

    logger.info("fitting the model")
    #experiment_id = mlflow.create_experiment(
    #    f"mart-surv-47"
    #)
    params = model_utils.GRUparams(epochs=1000, learn_rate=0.42678505501429896, hidden_dim=5, n_layers=1, batch_size=1024, drop_prob=0.15572706273371986)
    surv = survModel(params=params, feature_names=feature_names, scaler=scaler, category_encoder=category_encoder)
    surv.params.update({"epochs": 2000, "hidden_dim": 12,  "n_layers": 1, 'learn_rate': 0.25678505501429896})
    surv.fit(X=X_train, y=y_train, tune_hyperparams=False)

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

        y_pred = surv.predict(X_test)
        y_pred = surv.future_lifetime_quantiles(X_test)
        y_pred = surv.future_expected_lifetime(X_test, days_since_event=np.ones(len(X_test)))
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