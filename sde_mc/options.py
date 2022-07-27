import numpy as np
import torch
from scipy.integrate import quad
from scipy.stats import norm, lognorm
from abc import ABC, abstractmethod


def bs_binary_aon(spot, strike, expiry, r, sigma):
    """Computes the true value of a binary asset-or-nothing option under Black-Scholes assumptions

    :param spot: float
        The spot price of the asset

    :param strike: float
        The strike price of the option

    :param expiry: float
        The time to maturity of the option

    :param r: float
        The risk-free rate

    :param sigma: float
        The volatility of the asset

    :return: float
        The value of the option
    """
    upper_limit = (np.log(spot / strike) + (r + 0.5 * sigma * sigma) * expiry) / (np.sqrt(expiry) * sigma)
    lower_limit = -np.inf
    value_integral = quad(lambda z: np.exp(- 0.5 * z * z), lower_limit, upper_limit)[0]
    return value_integral / np.sqrt(2 * np.pi)


def bs_call(spot, strike, expiry, r, sigma):
    """ Computes the true value of a European call option under Black-Scholes assumptions

    :param spot: float
        The spot price of the asset

    :param strike: float
        The strike price of the option

    :param expiry: float
        The time to maturity of the option

    :param r: float
        The risk-free rate

    :param sigma: float
        The volatility of the asset

    :return: float
        The value of the option
    """
    d1 = (np.log(spot / strike) + (r + sigma ** 2 / 2) * expiry) / (sigma * np.sqrt(expiry))
    d2 = d1 - sigma * np.sqrt(expiry)
    return spot * norm.cdf(d1) - strike * np.exp(-r * expiry) * norm.cdf(d2)


def merton_call(spot, strike, expiry, r, sigma, alpha, gamma, rate):
    """Computes the true value of a European call option under the Merton jump-diffusion model

    :param spot: float
        The spot price of the asset

    :param strike: float
        The strike price of the option

    :param expiry: float
        The time to maturity of the option

    :param r: float
        The risk-free rate

    :param sigma: float
        The volatility of the asset

    :param alpha: float
        The mean of the log-jumps

    :param gamma: float
        The standard deviation of the log-jumps

    :param rate: float
        The intensity of the jumps

    :return: float
        The value of the option
    """
    beta = np.exp(alpha + 0.5 * gamma * gamma) - 1
    partial_sum = 0
    for k in range(40):
        r_k = r - rate * beta + (k * np.log(beta+1)) / expiry
        sigma_k = np.sqrt(sigma ** 2 + (k * gamma ** 2) / expiry)
        k_fact = np.math.factorial(k)
        partial_sum += (np.exp(-(beta+1) * rate * expiry) * ((beta+1) * rate * expiry) ** k / k_fact) * \
             bs_call(spot, strike, expiry, r_k, sigma_k)
    return partial_sum


def bs_digital_call(spot, strike, expiry, r, sigma):
    """Computes the true value of a digital option under Black-Scholes assumptions

    :param spot: float
        The spot price of the asset

    :param strike: float
        The strike price of the option

    :param expiry: float
        The time to maturity of the option

    :param r: float
        The risk-free rate

    :param sigma: float
        The volatility of the asset

    :return: float
        The value of the option
    """
    mn = np.log(spot) + (r - 0.5 * sigma * sigma) * expiry
    sd = sigma * np.sqrt(expiry)
    return (1 - lognorm.cdf(strike, s=sd, scale=np.exp(mn))) * np.exp(-r * expiry)


def bs_asian_call(spot, strike, expiry, r, sigma):
    """Computes the true value of a geometric Asian call under Black-Scholes assumptions

    :param spot: float
        The spot price of the asset

    :param strike: float
        The strike price of the option

    :param expiry: float
        The time to maturity of the option

    :param r: float
        The risk-free rate

    :param sigma: float
        The volatility of the asset

    :return: float
        The value of the option
    """
    sig_g = sigma / np.sqrt(3)
    b = 0.5 * (r - 0.5 * sig_g ** 2)
    d1 = (np.log(spot / strike) + (b + sig_g ** 2 / 2) * expiry) / (sig_g * np.sqrt(expiry))
    d2 = d1 - sig_g * np.sqrt(expiry)
    return spot * np.exp((b - r) * expiry) * norm.cdf(d1) - strike * np.exp(-r * expiry) * norm.cdf(d2)


class Option(ABC):
    """Abstract base class for options"""

    def __init__(self, log=False, discount=1):
        """
        :param log: bool
            If true, takes exponential of terminal value before applying payoff
        """
        self.log = log
        self.discount = discount

    def transform(self, x):
        spot = torch.exp(x) if self.log else x
        return self.discount * spot

    @abstractmethod
    def payoff(self, x):
        pass

    def __call__(self, x):
        return self.payoff(self.transform(x))


class EuroCall(Option):
    """European call option"""

    def __init__(self, strike, log=False, discount=1):
        """
        :param strike: float,
            The strike price of the option

        :param log: bool
            If true, takes exponential of terminal value before applying payoff

        :param discount: float
            Multiplicative factor to apply before payoff
        """
        super(EuroCall, self).__init__(log, discount)
        self.strike = strike

    def payoff(self, x):
        x = x[:, 0]
        return torch.where(x > self.strike, x - self.strike,
                           torch.tensor(0., dtype=x.dtype, device=x.device))


class EuroPut(Option):
    """European put option"""

    def __init__(self, strike, log=False, discount=1):
        """
        :param strike: float,
            The strike price of the option

        :param log: bool
            If true, takes exponential of terminal value before applying payoff

        :param discount: float
            Multiplicative factor to apply before payoff
        """
        super(EuroPut, self).__init__(log, discount)
        self.strike = strike

    def payoff(self, x):
        x = x[:, 0]
        return torch.where(x < self.strike, self.strike - x,
                           torch.tensor(0., dtype=x.dtype, device=x.device))


class BinaryAoN(Option):
    """Binary asset-or-nothing option"""

    def __init__(self, strike, log=False, discount=1):
        """
        :param strike: float,
            The strike price of the option

        :param log: bool
            If true, takes exponential of terminal value before applying payoff

        :param discount: float
            Multiplicative factor to apply before payoff
        """
        super(BinaryAoN, self).__init__(log, discount)
        self.strike = strike

    def payoff(self, x):
        x = x[:, 0]
        return torch.where(x >= self.strike, x, torch.tensor(0, dtype=x.dtype, device=x.device))


class Basket(Option):
    """Basket option"""

    def __init__(self, strike, average_type='arithmetic', log=False, discount=1):
        assert average_type in ['arithmetic', 'geometric']
        super(Basket, self).__init__(log, discount)
        self.strike = strike
        self.average_type = average_type

    def payoff(self, x):
        spot = x.mean(1) if self.average_type == 'arithmetic' else torch.exp(torch.log(x).mean(1))
        return torch.where(spot > self.strike, spot - self.strike, torch.tensor(0., dtype=spot.dtype,
                                                                                device=spot.device))


class Rainbow(Option):
    """Rainbow option (call on max)"""

    def __init__(self, strike, log=False, discount=1):
        super(Rainbow, self).__init__(log, discount)
        self.strike = strike

    def payoff(self, x):
        spot = x.max(1).values
        return torch.where(spot > self.strike, spot - self.strike, torch.tensor(0., dtype=spot.dtype,
                                                                                device=spot.device))


class Digital(Option):
    """Digital option"""

    def __init__(self, strike, log=False, discount=1):
        super(Digital, self).__init__(log, discount)
        self.strike = strike

    def payoff(self, x):
        x = x[:, 0]
        return torch.where(x > self.strike, torch.ones_like(x), torch.zeros_like(x))


class AsianCall(Option):
    """Asian call option"""

    def __init__(self, time_interval, strike, log=False, discount=1):
        super().__init__(log, discount)
        self.time_interval = time_interval
        self.strike = strike

    def payoff(self, x):
        spot = torch.exp(x[:, 1] / self.time_interval) if self.log else x[:, 1] / self.time_interval
        return torch.where(spot > self.strike, spot - self.strike,
                           torch.tensor(0., dtype=spot.dtype, device=spot.device))


class HestonRainbow(Option):
    def __init__(self, strike, log=False, discount=1):
        super(HestonRainbow, self).__init__(log, discount)
        self.strike = strike

    def payoff(self, x):
        even_inds = torch.tensor([i for i in range(len(x[0])) if not i%2])
        x = torch.index_select(x, 1, even_inds)
        spot = x.max(1).values
        return torch.where(spot > self.strike, spot - self.strike, torch.tensor(0., dtype=spot.dtype,
                                                                                device=spot.device))


class BestOf(Option):
    def __init__(self, strike, log=False, discount=1):
        super(BestOf, self).__init__(log, discount)
        self.strike = strike

    def payoff(self, x):
        max_assets = torch.max(x, dim=1).values
        return torch.maximum(max_assets, self.strike * torch.ones_like(max_assets))


class ConstantShortRate:
    """Constant short rate discounter"""

    def __init__(self, r):
        """
        :param r: float,
            the risk-free rate
        """
        self.r = r

    def __call__(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        return torch.exp(-t * self.r)
