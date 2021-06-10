import numpy as np
import torch
from scipy.integrate import quad
from scipy.stats import norm
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


def bs_call(spot, strike, expiry, r, sigma):
    d1 = (np.log(spot / strike) + (r + sigma ** 2 / 2) * expiry) / (sigma * np.sqrt(expiry))
    d2 = d1 - sigma * np.sqrt(expiry)
    return spot * norm.cdf(d1) - strike * np.exp(-r * expiry) * norm.cdf(d2)


def merton_jump_call(spot, strike, expiry, r, sigma, mean_jump, std_jump, rate):
    partial_sum = 0
    for k in range(40):
        r_k = r - rate * (mean_jump - 1) + (k * np.log(mean_jump)) / expiry
        sigma_k = np.sqrt(sigma ** 2 + (k * std_jump ** 2) / expiry)
        k_fact = np.math.factorial(k)
        partial_sum += (np.exp(-mean_jump * rate * expiry) * (mean_jump * rate * expiry) ** k / (k_fact)) * \
             bs_call(spot, strike, expiry, r_k, sigma_k)
    return partial_sum


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


class Rainbow(Option):
    def __init__(self, strike, log=False):
        super(Rainbow, self).__init__(log)
        self.strike = strike

    def __call__(self, x):
        x = torch.exp(x) if self.log else x
        spot = x.max(1).values
        return torch.where(spot > self.strike, spot - self.strike, torch.tensor(0., dtype=spot.dtype,
                                                                                device=spot.device))


class HestonRainbow(Option):
    def __init__(self, strike, log=False):
        super(HestonRainbow, self).__init__(log)
        self.strike = strike

    def __call__(self, x):
        x = torch.exp(x) if self.log else x
        even_inds = torch.tensor([i for i in range(len(x[0])) if not i%2])
        x = torch.index_select(x, 1, even_inds)
        spot = x.max(1).values
        return torch.where(spot > self.strike, spot - self.strike, torch.tensor(0., dtype=spot.dtype,
                                                                                device=spot.device))


class ConstantShortRate:
    def __init__(self, r):
        self.r = r

    def __call__(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        return torch.exp(-t * self.r)
