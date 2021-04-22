import numpy as np
import torch
from scipy.integrate import quad


def aon_payoff(spot, strike):
    """
    Computes the payoff of a binary asset-or-nothing option

    :param spot: torch.tensor, spot prices of asset
    :param strike: float, strike price of option
    :return: torch.tensor, the payoffs
    """
    return torch.where(spot >= strike, spot, torch.tensor(0, dtype=spot.dtype, device=spot.device))


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


