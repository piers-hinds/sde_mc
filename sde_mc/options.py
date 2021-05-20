import numpy as np
import torch
from scipy.integrate import quad
from abc import ABC, abstractmethod


def aon_true(spot, strike, r, vol, time):
    """
    Computes the true value of a binary asset-or-nothing option under GBM

    :param spot: the spot price of the asset
    :param strike: the strike price of the option
    :param r: the risk-free rate
    :param vol: the volatility of the asset
    :param time: the time to maturity of the option
    :return: the value of the option
    """
    upper_limit = ( np.log(spot / strike) + (r + 0.5 * vol * vol) * time ) / (np.sqrt(time) * vol)
    lower_limit = -np.inf
    value_integral = quad(lambda z: np.exp(- 0.5 * z * z), lower_limit, upper_limit)[0]
    return value_integral / np.sqrt(2 * np.pi)


class Option(ABC):
    """Abstract base class for options"""
    def __init__(self, log=False):
        self.log = log

    @abstractmethod
    def __call__(self, x):
        pass


class EuroCall(Option):
    """European call option"""

    def __init__(self, strike, log=False):
        super(EuroCall, self).__init__(log)
        self.strike = strike

    def __call__(self, x):
        spot = torch.exp(x[:, 0]) if self.log else x[:, 0]
        return torch.where(spot > self.strike, spot - self.strike,
                           torch.tensor(0., dtype=spot.dtype, device=spot.device))


class BinaryAoN(Option):
    """Binary asset-or-nothing option"""

    def __init__(self, strike, log=False):
        super(BinaryAoN, self).__init__(log)
        self.strike = strike

    def __call__(self, x):
        spot = torch.exp(x[:, 0]) if self.log else x[:, 0]
        return torch.where(spot >= self.strike, spot, torch.tensor(0, dtype=spot.dtype, device=spot.device))


class Basket(Option):
    def __init__(self, strike, average_type='arithmetic', log=False):
        assert average_type in ['arithmetic', 'geometric']
        super(Basket, self).__init__(log)
        self.strike = strike
        self.average_type = average_type

    def __call__(self, x):
        x = torch.exp(x) if self.log else x
        spot = x.mean(1) if self.average_type == 'arithmetic' else torch.exp(torch.log(x).mean(1))
        return torch.where(spot > self.strike, spot - self.strike, torch.tensor(0., dtype=spot.dtype,
                                                                                device=spot.device))


class ConstantShortRate:
    def __init__(self, r):
        self.r = r

    def __call__(self, t):
        return torch.exp(-t * self.r)
