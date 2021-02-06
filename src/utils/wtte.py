"""
Wrapper for Python Weibull functions
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Tuple

def dur_model_target(pd_y_train: pd.DataFrame):
    from lifelines import WeibullAFTFitter

    y_train_lifelines = pd.DataFrame()
    y_train_lifelines["duration"] = np.where(
        pd_y_train["label_lower_bound"] == pd_y_train["label_upper_bound"],
        pd_y_train["label_lower_bound"],
        pd_y_train["label_lower_bound"],
    )
    y_train_lifelines["event"] = np.where(
        pd_y_train["label_lower_bound"] == pd_y_train["label_upper_bound"], 1, 0
    )
    y_train_lifelines["duration"] = y_train_lifelines.where(y_train_lifelines["duration"] >= 1, 1)
    dur_model = WeibullAFTFitter(penalizer=1.5e-3, l1_ratio=1.0).fit(
        y_train_lifelines, duration_col="duration", event_col="event"
    )
    y_target_pred = dur_model.predict_median(
        y_train_lifelines, conditional_after=y_train_lifelines["duration"]
    )
    y_target_comb = pd.Series(
        np.where(
            y_train_lifelines["event"].astype(bool), y_train_lifelines["duration"], y_target_pred
        )
    )
    return y_target_comb

def weibull_cumulative_hazard(t, a, b):
    """ Cumulative hazard
    :param t: Value
    :param a: Alpha
    :param b: Beta
    :return: `np.power(t / a, b)`
    """
    t = np.double(t)
    return np.power(t / a, b)


def weibull_hazard(t, a, b):
    t = np.double(t)
    return (b / a) * np.power(t / a, b - 1)


def weibull_cdf(t, a, b):
    """ Cumulative distribution function.
    :param t: Value
    :param a: Alpha
    :param b: Beta
    :return: `1 - np.exp(-np.power(t / a, b))`
    """
    t = np.double(t)
    return 1 - np.exp(-np.power(t / a, b))


def weibull_pdf(t, a, b):
    """ Probability distribution function.
    :param t: Value
    :param a: Alpha
    :param b: Beta
    :return: `(b / a) * np.power(t / a, b - 1) * np.exp(-np.power(t / a, b))`
    """
    t = np.double(t)
    return (b / a) * np.power(t / a, b - 1) * np.exp(-np.power(t / a, b))


def weibull_cmf(t, a, b):
    """ Cumulative Mass Function.
    :param t: Value
    :param a: Alpha
    :param b: Beta
    :return: `cdf(t + 1, a, b)`
    """
    t = np.double(t) + 1e-35
    return weibull_cdf(t + 1, a, b)


def weibull_pmf(t, a, b):
    """ Probability mass function.
    :param t: Value
    :param a: Alpha
    :param b: Beta
    :return: `cdf(t + 1.0, a, b) - cdf(t, a, b)`
    """
    t = np.double(t) + 1e-35
    return weibull_cdf(t + 1.0, a, b) - weibull_cdf(t, a, b)


def weibull_mode(a, b):
    # Continuous mode.
    try:
        mode = a * np.power((b - 1.0) / b, 1.0 / b)
        mode[b <= 1.0] = 0.0
    except:
        # scalar case
        if b <= 1.0:
            mode = 0
        else:
            mode = a * np.power((b - 1.0) / b, 1.0 / b)
    return mode


def weibull_quantiles(a, b, p):
    """ Quantiles
    :param a: Alpha
    :param b: Beta
    :param p:
    :return: `a * np.power(-np.log(1.0 - p), 1.0 / b)`
    """
    return a * np.power(-np.log(1.0 - p), 1.0 / b)


def weibull_mean(a, b):
    """Continuous mean. Theoretically at most 1 step below discretized mean
    `E[T ] <= E[Td] + 1` true for positive distributions.
    :param a: Alpha
    :param b: Beta
    :return: `a * gamma(1.0 + 1.0 / b)`
    """
    from scipy.special import gamma
    return a * gamma(1.0 + 1.0 / b)


def weibull_continuous_loglik(t, a, b, u=1, equality=False):
    """Continous censored loglikelihood function.
    :param bool equality: In ML we usually only care about the likelihood
    with *proportionality*, removing terms not dependent on the parameters.
    If this is set to `True` we keep those terms.
    """
    if equality:
        loglik = u * np.log(weibull_pdf(t, a, b)) + (1 - u) * \
            np.log(1.0 - weibull_cdf(t, a, b))
    else:
        # commonly optimized over: proportional terms w.r.t alpha,beta
        loglik = u * loglik(weibull_hazard(t, a, b)) - \
            loglik(weibull_cumulative_hazard(t, a, b))

    return loglik


def weibull_discrete_loglik(t, a, b, u=1, equality=False):
    """Discrete censored loglikelihood function.
    :param bool equality: In ML we usually only care about the likelihood
    with *proportionality*, removing terms not dependent on the parameters.
    If this is set to `True` we keep those terms.
    """
    if equality:
        # With equality instead of proportionality.
        loglik = u * np.log(weibull_pmf(t, a, b)) + (1 - u) * \
            np.log(1.0 - weibull_cdf(t + 1.0, a, b))
    else:
        # commonly optimized over: proportional terms w.r.t alpha,beta
        hazard0 = weibull_cumulative_hazard(t, a, b)
        hazard1 = weibull_cumulative_hazard(t + 1., a, b)
        loglik = u * np.log(np.exp(hazard1 - hazard0) - 1.0) - hazard1

    return loglik

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


#class Loss(object):
#    """ Creates a keras WTTE-loss function.
#        - Usage
#            :Example:
#            .. code-block:: python
#               loss = wtte.Loss(kind='discrete').loss_function
#               model.compile(loss=loss, optimizer=RMSprop(lr=0.01))
#               # And with masking:
#               loss = wtte.Loss(kind='discrete',reduce_loss=False).loss_function
#               model.compile(loss=loss, optimizer=RMSprop(lr=0.01),
#                              sample_weight_mode='temporal')
#        .. note::
#            With masking keras needs to access each loss-contribution individually.
#            Therefore we do not sum/reduce down to scalar (dim 1), instead return a 
#            tensor (with reduce_loss=False).
#        :param kind:  One of 'discrete' or 'continuous'
#        :param reduce_loss: 
#        :param clip_prob: Clip likelihood to [log(clip_prob),log(1-clip_prob)]
#        :param regularize: Deprecated.
#        :param location: Deprecated.
#        :param growth: Deprecated.
#        :type reduce_loss: Boolean
#    """
#
#    def __init__(self,
#                 kind,
#                 reduce_loss=True,
#                 clip_prob=1e-6,
#                 regularize=False,
#                 location=None,
#                 growth=None):
#
#        self.kind = kind
#        self.reduce_loss = reduce_loss
#        self.clip_prob = clip_prob
#
#        if regularize == True or location is not None or growth is not None:
#            raise DeprecationWarning('Directly penalizing beta has been found \
#                                      to be unneccessary when using bounded activation \
#                                      and clipping of log-likelihood.\
#                                      Use this method instead.')
#
#    def loglik_discrete(self, y, u, a, b, epsilon=K.epsilon()):
#        hazard0 = K.pow((y + epsilon) / a, b)
#        hazard1 = K.pow((y + 1.0) / a, b)
#
#        loglikelihoods = u * \
#            K.log(K.exp(hazard1 - hazard0) - (1.0 - epsilon)) - hazard1
#        return loglikelihoods
#
#
#    def loglik_continuous(self, y, u, a, b, epsilon=K.epsilon()):
#        ya = (y + epsilon) / a
#        loglikelihoods = u * (K.log(b) + b * K.log(ya)) - K.pow(ya, b)
#        return loglikelihoods
#
#
#    def loss_function(self, y_true, y_pred):
#
#        y, u, a, b = _keras_split(y_true, y_pred)
#        if self.kind == 'discrete':
#            loglikelihoods = self.loglik_discrete(y, u, a, b)
#        elif self.kind == 'continuous':
#            loglikelihoods = self.loglik_continuous(y, u, a, b)
#
#        if self.clip_prob is not None:
#            loglikelihoods = K.clip(loglikelihoods, 
#                log(self.clip_prob), log(1 - self.clip_prob))
#        if self.reduce_loss:
#            loss = -1.0 * K.mean(loglikelihoods, axis=-1)
#        else:
#            loss = -loglikelihoods
#
#        return loss