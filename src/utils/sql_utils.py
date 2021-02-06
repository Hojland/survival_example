"""This module contains utils functions for data.
"""
import os
import sys
import getpass
import pandas as pd
from tqdm import tqdm
import sqlalchemy
import logging
from utils import utils


def load_data(engine, sql_query: str):
    df_load = pd.read_sql(sql_query, engine, chunksize=20000)
    try:
        df = pd.concat(
            [chunk for chunk in tqdm(df_load, desc="Loading data", file=sys.stdout)],
            ignore_index=True,
        )
    except ValueError:
        logging.error("No data in sql query table")
        df = pd.DataFrame()
    return df


def create_engine(host, port, db, user, pwd):
    """Creates a sqlalchemy engine, with specified connection information
    Arguments
    ---------
    host: string
       Host adress
    port: string
       Port for the host adress
    db: string
       Database name
    user: string
       User for connecting to database
    pwd: string
        Password for the provided user
    Returns
    -------
    engine: sqlalchemy Engine
    """
    engine = sqlalchemy.create_engine(
        "mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8".format(user, pwd, host, port, db)
    )
    return engine


def get_experiment_tag(db_engine: sqlalchemy.engine):
    experiment_tag = db_engine.execute(
        """
        SELECT MAX(experiment_id)
        FROM models.experiments
        """
    ).scalar()
    experiment_tag = int(experiment_tag)
    return experiment_tag


def table_exists(db_engine: sqlalchemy.engine, schema: str, table: str):
    exists_num = db_engine.execute(
        f"""
    SELECT EXISTS (SELECT * 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = '{schema}' 
        AND  TABLE_NAME = '{table}')
    """
    ).scalar()
    if exists_num == 0:
        exists = False
    elif exists_num == 1:
        exists = True
    return exists


def table_empty(db_engine: sqlalchemy.engine, table_name: str):
    empty_num = db_engine.execute(
        f"""
      SELECT EXISTS(SELECT 1 FROM {table_name})
   """
    ).scalar()
    if empty_num == 0:
        empty = False
    elif empty_num == 1:
        empty = True
    return empty


def table_exists_notempty(db_engine: sqlalchemy.engine, schema_name: str, table_name: str):
    exists = table_exists(db_engine, schema_name, table_name)
    if exists:
        empty = table_empty(db_engine, f"{schema_name}.{table_name}")
        if empty:
            both = True
        else:
            both = False
    else:
        both = False
    return both


def get_dtype_trans(df: pd.DataFrame, str_len: int = 150):
    obj_vars = [colname for colname in list(df) if df[colname].dtype == "object"]
    int_vars = [colname for colname in list(df) if df[colname].dtype == "int64"]
    float_vars = [
        colname
        for colname in list(df)
        if df[colname].dtype == "float64" or df[colname].dtype == "float32"
    ]
    date_vars = [colname for colname in list(df) if df[colname].dtype == "datetime64[ns]"]

    dtype_trans = {obj_var: sqlalchemy.String(str_len) for obj_var in obj_vars}
    dtype_trans.update({int_var: sqlalchemy.Integer for int_var in int_vars})
    dtype_trans.update({float_var: sqlalchemy.Float(14, 5) for float_var in float_vars})
    dtype_trans.update({date_var: sqlalchemy.Date for date_var in date_vars})
    return dtype_trans


def get_dtype_trans_notalchemy(df: pd.DataFrame, str_len: int = 150):
    obj_vars = [colname for colname in list(df) if df[colname].dtype == "object"]
    int_vars = [colname for colname in list(df) if df[colname].dtype == "int64"]
    float_vars = [
        colname
        for colname in list(df)
        if df[colname].dtype == "float64" or df[colname].dtype == "float32"
    ]
    date_vars = [colname for colname in list(df) if df[colname].dtype == "datetime64[ns]"]

    dtype_trans = {obj_var: f"VARCHAR({str_len})" for obj_var in obj_vars}
    dtype_trans.update({int_var: "INT" for int_var in int_vars})
    dtype_trans.update({float_var: "FLOAT(14, 5)" for float_var in float_vars})
    dtype_trans.update({date_var: "DATE" for date_var in date_vars})
    return dtype_trans


def create_table(
    mariadb_engine: sqlalchemy.engine,
    table_name: str,
    col_datatype_dct: dict,
    primary_key: str = None,
    index_lst: list = None,
    foreignkey_ref_dct: dict = None,
):
    # primary_key = "id INT AUTO_INCREMENT PRIMARY KEY"
    def_strings = []
    col_definition_str = ", ".join([f"{k} {v}" for k, v in col_datatype_dct.items()])
    if primary_key:
        col_definition_str = primary_key + ", " + col_definition_str
    def_strings.append(col_definition_str)
    if foreignkey_ref_dct:
        foreign_key_strs = [
            f"FOREIGN KEY ({k}) REFERENCES {v}" for k, v in foreignkey_ref_dct.items()
        ]
        foreign_str = ", ".join(foreign_key_strs)
        def_strings.append(foreign_str)
    if index_lst:
        index_str = ", ".join([f"INDEX ({index})" for index in index_lst])
        def_strings.append(index_str)

    create_table_query = f"""CREATE TABLE IF NOT EXISTS {table_name} ({','.join(def_strings)});"""

    mariadb_engine.execute(create_table_query)


def df_to_sql_split(
    mariadb_engine: sqlalchemy.engine, df: pd.DataFrame, table_name: str, chunksize: int = 50
):
    for i in range(0, len(df), chunksize):
        df_to_sql(mariadb_engine, df.iloc[i : i + chunksize], table_name)


def df_to_sql(mariadb_engine: sqlalchemy.engine, df: pd.DataFrame, table_name: str):
    def delete_quotation(string: str):
        return string.replace("'", "").replace('"', "")

    df = df.astype(str)
    df_values = [[delete_quotation(value) for value in values] for values in df.values]
    sql_query_start = f"INSERT INTO {table_name}"
    column_str = ",".join(list(df))
    values_str = ",".join([f"""('{"','".join(values)}')""" for values in df_values])
    values_str = general_utils.multiple_replace({"'nan'": "NULL", "'<NA>'": "NULL"}, values_str)

    sql_query = f"{sql_query_start} ({column_str}) VALUES {values_str}"
    mariadb_engine.execute(sql_query)