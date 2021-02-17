#%load_ext autoreload
#%autoreload 2
import pandas as pd
import numpy as np
import mlflow
import re
import os
import shap
import matplotlib.pyplot as plt
import sqlalchemy
import logging

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
    df = get_data()
    logger.info("loading the model")

    df = df.set_index(['customerID'])

    X, y = preprocessing_utils.split_X_y(df, y_variables=['not_censored', 't'])
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = preprocessing_utils.train_test_split(X, y, settings.TEST_SIZE, settings.SEED)
    X_train, X_test, y_train, y_test = preprocessing_utils.to_cubes(X_train, X_test, y_train, y_test, max_seq_len=2)

    #from utils import wtte_torch
    #alpha, beta = wtte_torch.weibull_baseline(t=y_train[:, 1, 1], u=y_train[:, 1, 0])

    #wtte_torch.weibull_mean(alpha, beta)
    #wtte_torch.plot_weibull_pdf(alpha, beta)

    #logger.info("fitting the model")
    #experiment_id = mlflow.create_experiment(
    #    f"mart-surv-3"
    #)
    #surv = survModel()
    #surv.fit(X=X_train, y=y_train, tune_params=False, experiment_id=experiment_id)

    logger.info("refitting the model to best params")
    with mlflow.start_run(experiment_id=experiment_id):
        os.makedirs("tmp_artifacts", exist_ok=True)
        surv.fit(X=x_train, y=y_train, tune_params=False)
        mlflow.set_tags({"lifecycle": "FINAL_RUN"})
        mlflow.log_params(surv.params)
        fig = surv.explain(x_test)
        np.save("tmp_artifacts/shap_base_values", surv.explainer.expected_value)
        mlflow.log_artifact(local_path="tmp_artifacts/shap_base_values.npy",)
        np.save("tmp_artifacts/shap_values", surv.shap_values)
        mlflow.log_artifact(local_path="tmp_artifacts/shap_values.npy")

        surv.shap_summary(x_test, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig("tmp_artifacts/shap_fig.svg")
        mlflow.log_artifact(local_path="tmp_artifacts/shap_fig.svg")

        y_pred = surv.predict(x_test)

        # TODO get metrics
        mlflow.log_metrics(res)

        xgboo.log_model(
            artifact_path=f"obscure/{MODEL_NAME}",
            registered_model_name=MODEL_NAME,
        )
        os.system("rm -rf tmp_artifacts")

    # TODO implement the eval loss
    #res["eval_loss"] = surv.eval_loss(x_test, y_test)

    res.update(surv.params)
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
# and minmax scaler to model
# add model training to model class
# add hyperopt to hyperparameters
# implement SHAP