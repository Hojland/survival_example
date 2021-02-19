"""
Wrapper for Python Weibull functions
"""
import numpy as np
import pandas as pd
import torch
from typing import Tuple

class torchWeibullLoss:
    """ Creates a Pytorch WTTE-loss function.
    """
    def __init__(self,
                 reduce_loss=True,
                 clip_prob=1e-6,
                 regularize=False,
                 location=None,
                 growth=None):   
        self.reduce_loss = reduce_loss
        self.clip_prob = clip_prob   
        if regularize == True or location is not None or growth is not None:
            raise DeprecationWarning('Directly penalizing beta has been found \
                                      to be unneccessary when using bounded activation \
                                      and clipping of log-likelihood.\
                                      Use this method instead.')

    def loglik_continuous(self, t, u, a, b, epsilon=1e-10):
        """ Cumulative hazard
        :param t: time
        :param u: non_censoring
        :param a: alpha
        :param b: beta
        :return: loglikelihood vector
        """
        ta = (t + epsilon) / a
        loglikelihoods = u * (torch.log(b) + b * torch.log(ta)) - ta ** b
        return loglikelihoods

    def extract_parameters(self, output):
        def ensure_pos(param, eps=.1e-4):
            param = torch.where(param > 0, param, torch.tensor([eps]))
            return param
        #a = ensure_pos(output[:, 0])
        a = output[:, 0]
        #b = ensure_pos(output[:, 1])
        b = output[:, 1]
        return a, b

    def extract_cens_time(self, target):
        t = target[:, 1]
        u = target[:, 0]
        return t, u

    def loss(self, output, target):
        seq_len = output.shape[1]
        loglikelihoods = torch.Tensor()
        a, b = self.extract_parameters(output)
        for seq in range(seq_len):
            t, u = self.extract_cens_time(target[:, seq, :])
            loglikelihoods_seq = self.loglik_continuous(t, u, a, b)
            loglikelihoods = torch.cat((loglikelihoods, loglikelihoods_seq), 0)
        if self.clip_prob is not None:
            loglikelihoods = torch.clip(loglikelihoods, 
                np.log(self.clip_prob), np.log(1 - self.clip_prob))

        if self.reduce_loss:
            loglik = - 1.0 * torch.mean(loglikelihoods)
        else:
            loglik = - loglikelihoods
        return loglik

# should these be torch functions?
def weibull_expected_future_lifetime(t0, a, b):
    def wolfram_inc_gamma(arg1, arg2):
        from scipy.special import gammaincc, gamma
        return gamma(arg1)*gammaincc(arg1, arg2)
    cum_hazard_t0 = weibull_cumulative_hazard(t0, a, b)
    exp_lifetime = (t0 * np.power(cum_hazard_t0, -1/b)) / ( np.exp(-cum_hazard_t0) * b) * wolfram_inc_gamma(
        1 / b, cum_hazard_t0
    )
    return exp_lifetime

def weibull_future_lifetime_quantiles(q, t0, a, b):
    q_lifetime = a * (((t0 / a) ** b) - np.log(q)) ** (1 / b) - t0
    return q_lifetime

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

def plot_weibull_pdf(alpha, beta, max_t: int=200):
    pdf = []
    for t in range(200):
        pdf.append(weibull_pdf(t, alpha, beta))
    pdf = pd.Series(pdf)
    pdf.plot()

def weibull_baseline(t: np.ndarray, u: np.ndarray):
    from lifelines import WeibullFitter
    wbf = WeibullFitter()
    wbf.fit(t, u)
    wbf.plot()
    alpha, beta = wbf.lambda_, wbf.rho_
    return alpha, beta

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