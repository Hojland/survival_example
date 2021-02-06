import os
from datetime import timedelta
from typing import List
from pydantic import (
    BaseSettings,
    AnyHttpUrl,
    SecretStr,
    HttpUrl,
)
import pytz

LOCAL_TZ = pytz.timezone("Europe/Copenhagen")

MLFLOW_TRACKING_USERNAME: SecretStr="MLFLOW_USERNAME"
MLFLOW_TRACKING_PASSWORD: SecretStr="MLFLOW_PSW"

SQL_SCHEMA = "output"
SQL_TABLE_NAME = "models"
MODEL_NAME = "surv-model"
MODEL_STAGE = "production"
MLFLOW_URI = "https://mlflow.nuuday-ai.cloud/"
TEST_SIZE=0.25
SEED=42