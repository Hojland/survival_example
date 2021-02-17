"""
Wrapper for Python Weibull functions
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Tuple


def weibull_timevaryingcov_continuous_gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient for weibull with timevarying covariates.'''
    d_alpha = 0
    d_beta = 0
    for id in np.unique(dtrain.index):
        id_censor = dtrain.censored[dtrain.index==id][len(dtrain.censored[dtrain.index==id])-1]
        if True:
            break
        id_start = dtrain.start[dtrain.index==id]
        id_stop = dtrain.stop[dtrain.index==id]

        alpha = predt[dtrain.index==id][0]
        beta = predt[dtrain.index==id][1]
        d_alpha = d_alpha + -id_censor * beta * np.power(alpha, -1) \
            + np.sum(beta * np.power(id_stop, beta) * np.power(alpha, -beta-1) \
            - beta * np.power(id_start, beta) * np.power(alpha, -beta-1))
        d_beta = d_beta + id_censor * (np.log(np.max(id_stop)/alpha) + np.power(beta, -1)) \
            - np.sum(np.log(id_stop/alpha) * np.power(alpha, -beta) * np.power(id_stop, beta) \
            - np.log(id_start/alpha) * np.power(alpha, -beta) * np.power(id_start, beta))
    
    grad = np.array([d_alpha / len(np.unique(dtrain.index)), d_beta / len(np.unique(dtrain.index))])
    return grad

def weibull_timevaryingcov_continuous_hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for weibull with timevarying covariates.'''
    dd_alpha_alpha = 0
    dd_beta_beta = 0
    dd_alpha_beta = 0
    for id in np.unique(dtrain.index):
        id_censor = dtrain.censored[dtrain.index==id][len(dtrain.censored[dtrain.index==id])-1]
        id_start = dtrain.start[dtrain.index==id]
        id_stop = dtrain.stop[dtrain.index==id]

        alpha = predt[dtrain.index==id][0]
        beta = predt[dtrain.index==id][1]

        dd_alpha_alpha = dd_alpha_alpha + id_censor * np.power(alpha, -2) * beta - \
            np.sum(beta * (beta + 1) * np.power(id_stop, beta) * np.power(alpha, -beta-2) \
            - beta * (beta + 1) * np.power(id_start, beta) * np.power(alpha, -beta-2))
        
        dd_beta_beta = dd_beta_beta - id_censor * np.power(beta, -2) - \
            np.sum(np.power(np.log(id_stop/alpha), 2) * np.power(alpha, -beta) * np.power(id_stop, beta) \
            - np.power(np.log(id_start/alpha), 2) * np.power(alpha, -beta) * np.power(id_start, beta))
        
        dd_alpha_beta = dd_alpha_beta - id_censor * np.power(alpha, -1) + \
            np.sum(np.power(id_stop, beta) * np.power(alpha, -beta-1) + beta * np.power(alpha, -1) * np.log(id_stop/alpha) \
            - (np.power(id_start, beta) * np.power(alpha, -beta-1) + beta * np.power(alpha, -1) * np.log(id_start/alpha)))

    dd_alpha_alpha = dd_alpha_alpha / len(np.unique(dtrain.index))
    dd_beta_beta = dd_beta_beta / len(np.unique(dtrain.index))
    dd_alpha_beta = dd_alpha_beta / len(np.unique(dtrain.index))
    hess = np.array([[dd_alpha_alpha, dd_alpha_beta],[dd_alpha_beta, dd_beta_beta]])
    return hess

def weibull_timevaryingcov_continuous_obj(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Weibull timevarying covariates Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = weibull_timevaryingcov_continuous_gradient(predt, dtrain)
    hess = weibull_timevaryingcov_continuous_hessian(predt, dtrain)
    print(f"grad: {grad}")
    print(f"hess: {hess}")
    return grad, hess


def weibull_timevaryingcov_continuous_loglik_m(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    ''' Weibull timevarying covariates continuous loglikelihood error metric.'''
    predt[predt < -1] = -1 + 1e-6
    loglik = 0
    for id in np.unique(dtrain.index):
        id_censor = dtrain.censored[dtrain.index==id][len(dtrain.censored[dtrain.index==id])-1]
        id_start = dtrain.start[dtrain.index==id]
        id_stop = dtrain.stop[dtrain.index==id]

        alpha = predt[dtrain.index==id][0]
        beta = predt[dtrain.index==id][1]
        element_0 = beta * np.log(np.max(id_stop) / alpha) + np.log(beta)
        element_1 = np.sum(np.power(id_stop/alpha, beta)-np.power(id_start/alpha, beta))
        loglik = loglik + id_censor * element_0 + element_1
    return 'weibull_loglik', float(loglik / len(np.unique(dtrain.index)))