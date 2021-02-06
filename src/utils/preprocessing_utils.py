import pandas as pd
import numpy as np
import os
#from sklearn import model_selection

SEED = 42
TEST_SIZE = 0.25

def determine_cat(df: pd.DataFrame):
    cat = [col for col in df if df[col].dtype.name in ['object', 'category']]
    return cat

def downsample(df: pd.DataFrame, sample_frac: float = 1, seed: int = None):
    if seed is None:
        seed = SEED
    df = df.sample(frac=sample_frac, random_state=seed)
    return df

def train_test_split(X: pd.DataFrame, y: pd.DataFrame,
                     test_size: float=None, random_state: int=None):
    if test_size is None:        
        test_size = TEST_SIZE
    if random_state is None:
        random_state = SEED

    if 0 <= test_size <= 1: 
        test_size = int(np.floor(test_size * np.unique(X.index.values).shape[0]))

    ids = np.unique(X.index.values)
    test_idx = np.random.choice(ids, size=test_size, replace=False)
    train_idx = ids[~np.isin(ids, test_idx)]

    X_train, X_test, y_train, y_test = \
        X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]

    return X_train, X_test, y_train, y_test

def split_X_y(df: pd.DataFrame, y_variables: list):
    y = df[y_variables]
    X = df.drop(y_variables,  axis=1)
    return X, y


def preprocessing(df: pd.DataFrame, make_dummies: bool, scale: bool = True):
    cat_features = determine_cat(df)
    if make_dummies:
        df = pd.get_dummies(df, columns=cat_features)
    else:
        df[cat_features] = df[cat_features].fillna(value='Unknown')
        df[cat_features] = df[cat_features].astype('category')
    return df


def feature_engineering(df: pd.DataFrame):
    return df


def postprocessing(df: pd.DataFrame, scale: bool = True):
    return df
