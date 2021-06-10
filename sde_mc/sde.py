from abc import ABC, abstractmethod
import torch
import numpy as np
from .block_diag import block_diag


class Sde(ABC):
    """An abstract base class for SDEs"""

    def __init__(self, init_value, dim, noise_dim, corr_matrix=None):
        """
        :param init_value: torch.tensor, the initial value of the process
        :param dim: int, the dimension of the process
        :param noise_dim: int, dimension of the Wiener process
        :param corr_matrix: torch.tensor, the correlation matrix for the Wiener processes. If None, independence is
        assumed
        """
        self.init_value = init_value
        self.dim = dim
        self.noise_dim = noise_dim
        if corr_matrix is None:
            self.corr_matrix = torch.eye(self.noise_dim)
        else:
            self.corr_matrix = corr_matrix

    @abstractmethod
    def drift(self, t, x):
        """YOUR CODE HERE
        :param t: torch.tensor, the current time
        :param x: torch.tensor (bs, dim), the current value of the process
        :return: torch.tensor (bs, dim), the drift vector of the process at (t, x)
        """
        pass

    @abstractmethod
    def diffusion(self, t, x):
        """YOUR CODE HERE
        :param t: torch.tensor, the current time
        :param x: torch.tensor (bs, dim), the current value of the process
        :return: torch.tensor (bs, dim, noise_dim), the diffusion matrix of the process at (t, x)
        """
        pass


class SdeJumps(Sde):
    """Abstract class for SDEs with jumps"""

    def __init__(self, base_sde, rate):
        super(SdeJumps, self).__init__(base_sde.init_value, 1, 1, None)
        self.base_sde = base_sde
        self.rate = rate

    def drift(self, t, x):
        return self.base_sde.drift(t, x)

    def diffusion(self, t, x):
        return self.base_sde.diffusion(t, x)

    @abstractmethod
    def jumps(self, t, x):
        """The function coefficient of the compound Poisson process"""
        pass

    @abstractmethod
    def mean_jumps(self):
        """The mean of the jump sizes"""
        pass


class SdeLogNormalJumps(SdeJumps):
    def __init__(self, base_sde, rate, mean, std):
        super(SdeLogNormalJumps, self).__init__(base_sde, rate)
        self.mean = mean
        self.std = std

    @abstractmethod
    def jumps(self, t, x):
        pass

    def mean_jumps(self):
        return np.exp(self.mean + 0.5 * self.std * self.std) - 1


class Gbm(Sde):
    """Multi-dimensional GBM with possible correlation"""
    def __init__(self, mu, sigma, init_value, dim, corr_matrix=None):
        """
        :param mu: torch.tensor, the drifts of the process
        :param sigma: torch.tensor, the volatilities of the process
        :param init_value: torch.tensor, the initial value of the process
        :param dim: torch.tensor, the dimension of the GBM
        :param corr_matrix: torch.tensor, the correlation matrix
        """
        super(Gbm, self).__init__(init_value, dim, dim, corr_matrix)
        self.mu = mu
        self.sigma = sigma

    def drift(self, t, x):
        return self.mu * x

    def diffusion(self, t, x):
        return torch.diag_embed(self.sigma * x)


class Merton(SdeLogNormalJumps):
    """One-dimensional Merton jump-diffusion model"""
    def __init__(self, mu, sigma, init_value, rate, mean, std):
        """
        :param mu: torch.tensor, the drift of the process
        :param sigma: torch.tensor, the volatility of the process
        :param init_value: torch.tensor, the initial value of the process
        :param rate: torch.tensor, the rate of the Poisson process
        :param mean: torch.tensor, the mean of the jumps
        :param std: torch.tensor, the standard deviation of the jumps
        """
        super(Merton, self).__init__(Gbm(mu, sigma, init_value, 1), rate, mean, std)

    def jumps(self, t, x):
        return x


class Heston(Sde):
    """The (log) Heston model under EMM dynamics"""

    def __init__(self, r, kappa, theta, xi, rho, init_value):
        """
        :param r: float, the risk-free rate
        :param kappa: float, mean-reversion rate of the variance process
        :param theta: float, the long-run mean of the variance process
        :param xi: float, the vol-of-vol
        :param rho: float, the correlation between the two Wiener processes
        :param init_value: torch.tensor (2), the initial spot price and the initial variance
        """
        assert 2 * kappa * theta > xi**2, "Feller condition not satisfied"
        super(Heston, self).__init__(init_value=init_value, dim=2, noise_dim=2,
                                     corr_matrix=torch.tensor([[1., rho], [rho, 1.]]))
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi

    def drift(self, t, x):
        return torch.cat([self.r - 0.5 * torch.exp(x[:, 1]).unsqueeze(1),
                          ((self.kappa * (self.theta - torch.exp(x[:, 1])) - 0.5*self.xi**2) /
                           torch.exp(x[:, 1])).unsqueeze(1)], dim=1)

    def diffusion(self, t, x):
        return torch.diag_embed(torch.cat([torch.sqrt(torch.exp(x[:, 1])).unsqueeze(1),
                                           self.xi / torch.sqrt(torch.exp(x[:, 1])).unsqueeze(1)], dim=1))


class MultiHeston(Sde):
    """Multiple Heston models"""

    def __init__(self, r, kappa, theta, xi, rho, init_value, dim):
        """
        :param r: float, the risk-free rate
        :param kappa: float, mean-reversion rate of the variance process
        :param theta: float, the long-run mean of the variance process
        :param xi: float, the vol-of-vol
        :param rho: float, the correlation between the two Wiener processes
        :param init_value: torch.tensor (2), the initial spot price and the initial variance
        """
        assert torch.all(torch.gt(2 * kappa * theta, xi ** 2)), "Feller condition not satisfied"
        super(MultiHeston, self).__init__(init_value=init_value, dim=2*dim, noise_dim=2*dim,
                                          corr_matrix=torch.tensor([[1., rho[0], 0, 0], [rho[0], 1., 0, 0],
                                                                    [0, 0, 1, rho[1]], [0, 0, rho[1], 1]]))
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi

        self.h1 = Heston(r[0], kappa[0], theta[0], xi[0], rho[0], init_value[:2])
        self.h2 = Heston(r[1], kappa[1], theta[1], xi[1], rho[1], init_value[2:4])

    def drift(self, t, x):
        return torch.cat([self.h1.drift(t, x), self.h2.drift(t, x)], dim=-1)

    def diffusion(self, t, x):
        return block_diag([self.h1.diffusion(t, x), self.h2.diffusion(t, x)])
