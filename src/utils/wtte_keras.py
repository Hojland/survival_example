"""
Wrapper for Python Weibull functions
"""
import numpy as np
import pandas as pd
import keras as K
from typing import Tuple

class Loss(object):
    """ Creates a keras WTTE-loss function.
        - Usage
            :Example:
            .. code-block:: python
               loss = wtte.Loss(kind='discrete').loss_function
               model.compile(loss=loss, optimizer=RMSprop(lr=0.01))
               # And with masking:
               loss = wtte.Loss(kind='discrete',reduce_loss=False).loss_function
               model.compile(loss=loss, optimizer=RMSprop(lr=0.01),
                              sample_weight_mode='temporal')
        .. note::
            With masking keras needs to access each loss-contribution individually.
            Therefore we do not sum/reduce down to scalar (dim 1), instead return a 
            tensor (with reduce_loss=False).
        :param kind:  One of 'discrete' or 'continuous'
        :param reduce_loss: 
        :param clip_prob: Clip likelihood to [log(clip_prob),log(1-clip_prob)]
        :param regularize: Deprecated.
        :param location: Deprecated.
        :param growth: Deprecated.
        :type reduce_loss: Boolean
    """

    def __init__(self,
                 kind,
                 reduce_loss=True,
                 clip_prob=1e-6,
                 regularize=False,
                 location=None,
                 growth=None):

        self.kind = kind
        self.reduce_loss = reduce_loss
        self.clip_prob = clip_prob

        if regularize == True or location is not None or growth is not None:
            raise DeprecationWarning('Directly penalizing beta has been found \
                                      to be unneccessary when using bounded activation \
                                      and clipping of log-likelihood.\
                                      Use this method instead.')

    def loglik_discrete(self, y, u, a, b, epsilon=K.epsilon()):
        hazard0 = K.pow((y + epsilon) / a, b)
        hazard1 = K.pow((y + 1.0) / a, b)

        loglikelihoods = u * \
            K.log(K.exp(hazard1 - hazard0) - (1.0 - epsilon)) - hazard1
        return loglikelihoods


    def loglik_continuous(self, y, u, a, b, epsilon=K.epsilon()):
        ya = (y + epsilon) / a
        loglikelihoods = u * (K.log(b) + b * K.log(ya)) - K.pow(ya, b)
        return loglikelihoods


    def loss_function(self, y_true, y_pred):

        y, u, a, b = _keras_split(y_true, y_pred)
        if self.kind == 'discrete':
            loglikelihoods = self.loglik_discrete(y, u, a, b)
        elif self.kind == 'continuous':
            loglikelihoods = self.loglik_continuous(y, u, a, b)

        if self.clip_prob is not None:
            loglikelihoods = K.clip(loglikelihoods, 
                log(self.clip_prob), log(1 - self.clip_prob))
        if self.reduce_loss:
            loss = -1.0 * K.mean(loglikelihoods, axis=-1)
        else:
            loss = -loglikelihoods

        return loss