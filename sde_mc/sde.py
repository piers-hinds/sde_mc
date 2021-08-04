from abc import ABC, abstractmethod
import torch
import numpy as np


class Sde(ABC):
    """Abstract base class for SDEs driven by a Wiener process and a Poisson process"""

    def __init__(self, init_value, dim, corr_matrix=None, method='euler'):
        self.init_value = init_value
        self.dim = dim
        self.simulation_method = method
        if corr_matrix is not None:
            self.corr_matrix = corr_matrix
        else:
            self.corr_matrix = torch.eye(dim)

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
        :return: torch.tensor (bs, dim), the diffusion vector of the process at (t, x)
        """
        pass

    @abstractmethod
    def jumps(self, t, x):
        """YOUR CODE HERE
        :param t: torch.tensor, the current time
        :param x: torch.tensor (bs, dim), the current value of the process
        :return: torch.tensor (bs, dim), the jump coefficient of the process at (t, x)
        """
        pass

    @abstractmethod
    def sample_jumps(self, size, device):
        """YOUR CODE HERE
        :param size: tuple of ints, the size (shape) of the output of sampled jumps
        :param device: str, the device on which to sample the jumps
        :return: torch.tensor, sampled jumps
        """
        pass

    @abstractmethod
    def jump_mean(self):
        """YOUR CODE HERE
        :return: float, the mean of the jumps
        """
        pass

    @abstractmethod
    def jump_rate(self):
        """YOUR CODE HERE
        :return: float, the rate of the Poisson process
        """
        pass


class DiffusionSde(Sde):
    """Abstract class for SDEs with only a drift and diffusion component"""
    @abstractmethod
    def drift(self, t, x):
        pass

    @abstractmethod
    def diffusion(self, t, x):
        pass

    def jumps(self, t, x):
        return None

    def sample_jumps(self, size, device):
        return None

    def jump_mean(self):
        return None

    def jump_rate(self):
        return 0


class Gbm(DiffusionSde):
    """Multi-dimensional GBM with possible correlation"""

    def __init__(self, mu, sigma, init_value, dim, corr_matrix=None, method='euler'):
        """
        :param mu: torch.tensor, the drifts of the process
        :param sigma: torch.tensor, the volatilities of the process
        :param init_value: torch.tensor, the initial value of the process
        :param dim: torch.tensor, the dimension of the GBM
        :param corr_matrix: torch.tensor, the correlation matrix
        """
        super(Gbm, self).__init__(init_value, dim, corr_matrix, method)
        self.mu = mu
        self.sigma = sigma

    def drift(self, t, x):
        return self.mu * x

    def diffusion(self, t, x):
        return self.sigma * x
    

class Heston(DiffusionSde):
    def __init__(self, r, kappa, theta, xi, rho, init_value):
        assert 2 * kappa * theta > xi**2
        corr_matrix = torch.tensor([[1., rho], [rho, 1.]])
        super(Heston, self).__init__(init_value, 2, corr_matrix)
        self.simulation_method = 'heston'
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi

    def drift(self, t, x):
        return torch.stack([r * x[:, 0], torch.zeros_like(x[:, 1])], dim=1)

    def diffusion(self, t, x):
        return torch.stack([(torch.sqrt(x[:, 1]) * x[:, 0]), torch.zeros_like(x[:, 1])], dim=1)

    def quadratic_parameters(self, x, h, normals):
        a = - torch.ones_like(x) - self.kappa * h
        b = self.xi * normals
        c = x + self.kappa * self.theta * h - 0.5 * h * self.xi**2
        return a, b, c


class LogNormalJumpsSde(Sde):
    """SDE with jumps that have shifted log-normal distribution"""

    def __init__(self, rate, alpha, gamma, init_value, dim, corr_matrix=None, method='euler'):
        """
        :param rate: float, the rate of the Poisson process
        :param alpha: float, the mean of the jumps
        :param gamma: float, the standard deviation of the jumps
        :param init_value: torch.tensor, the initial value of the process
        :param dim: int, the dimension of the SDE
        :param corr_matrix: torch.tensor, the correlation matrix
        """
        super(LogNormalJumpsSde, self).__init__(init_value, dim, corr_matrix, method)
        self.rate = rate
        self.alpha = alpha
        self.gamma = gamma

    @abstractmethod
    def drift(self, t, x):
        pass

    @abstractmethod
    def diffusion(self, t, x):
        pass

    @abstractmethod
    def jumps(self, t, x):
        pass

    def sample_jumps(self, size, device):
        return (torch.randn(size=size, device=device) * self.gamma + self.alpha).exp() - 1

    def jump_mean(self):
        return np.exp(self.alpha + 0.5 * self.gamma * self.gamma) - 1

    def jump_rate(self):
        return self.rate


class Merton(LogNormalJumpsSde):
    """Merton jump-diffusion model (GBM with shifted log-normal jumps)"""

    def __init__(self, mu, sigma, rate, alpha, gamma, init_value, dim, corr_matrix=None, method='euler'):
        """
        :param mu: float, the drift of the process
        :param sigma: float, the volatility of the process
        :param rate: float, the rate of the Poisson process
        :param alpha: float, the mean of the jumps
        :param gamma: float, the standard deviation of the jumps
        :param init_value: torch.tensor, the initial value of the process
        :param dim: int, the dimension of the SDE
        :param corr_matrix: torch.tensor, the correlation matrix
        """
        self.mu = mu
        self.sigma = sigma
        super(Merton, self).__init__(rate, alpha, gamma, init_value, dim, corr_matrix, method)

    def drift(self, t, x):
        return (self.mu - self.rate * self.jump_mean()) * x

    def diffusion(self, t, x):
        return self.sigma * x

    def jumps(self, t, x):
        return x

