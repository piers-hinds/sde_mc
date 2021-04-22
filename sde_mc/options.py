import numpy as np
from scipy.integrate import quad

def aon_payoff(spot, strike):
    """
    Computes the payoff of a binary asset-or-nothing option

    :param spot: the spot price of the asset
    :param strike: the strike price of the option
    :return: payoff
    """
    if spot >= strike:
        return spot
    return 0.0

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


