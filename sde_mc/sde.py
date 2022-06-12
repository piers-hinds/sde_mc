from abc import ABC, abstractmethod
import torch
import numpy as np


class Sde(ABC):
    """Abstract base class for SDEs driven by a Brownian motion and a Poisson process.

    Each Sde object needs the following attributes:

        init_value:
            The initial value of the SDE

        dim:
            The dimension of the solution to the SDE

        brown_dim:
            The dimension of the Brownian motion driving the SDE

        diffusion_struct:
            The type of structure of the diffusion matrix of the SDE. In many cases the general setting of a
            (dim, brown_dim)-shaped diffusion matrix is not efficient for simulation. Possible types include:

                'diag': For when the diffusion matrix is diagonal (the brown_dim = dim). The output of Sde.diffusion()
                    will be (dim,)-shaped

                'indep': For when the diffusion matrix has two or more diagonal stripes corresponding to when the SDE is
                    driven by two or more independent dim-dimensional brownian motions. The output of Sde.diffusion()
                    will be (dim, brown_dim / dim)-shaped

                'general': For the general case when there is no structure of the diffusion matrix. The shape of
                    Sde.diffusion() will be (dim, brown_dim)

        corr_matrix:
            The correlation matrix for the Brownian motion. Defaults to the identity matrix.

        method:
            The preferred simulation method for the SDE. Defaults to Euler method and can be overridden when solving.
    """

    def __init__(self, init_value, dim, brown_dim, diffusion_struct, corr_matrix=None, method='euler'):
        """
        :param init_value: torch.tensor
            The initial value of the SDE. Should be (dim,)-shaped

        :param dim: int
            The dimension of the solution to the SDE

        :param brown_dim: int
            The dimension of the Brownian motion driving the SDE

        :param diffusion_struct: str
            The type of structure of the diffusion matrix

        :param corr_matrix: torch.tensor
            The correlation matrix of the Brownian motion. Should be (brown_dim, brown_dim)-shaped

        :param method: str
            The preferred simulation method
        """
        self.init_value = init_value
        self.dim = dim
        self.diffusion_struct = diffusion_struct
        self.brown_dim = brown_dim
        self.simulation_method = method
        if corr_matrix is not None:
            self.corr_matrix = corr_matrix
        else:
            self.corr_matrix = torch.eye(dim)

    @abstractmethod
    def drift(self, t, x):
        """The drift of the SDE at time-space (t, x)

        :param t: torch.tensor
            The current time

        :param x: torch.tensor (bs, dim)
            The current value of the process

        :return: torch.tensor (bs, dim)
            The drift vector of the process at (t, x)
        """
        pass

    @abstractmethod
    def diffusion(self, t, x):
        """The diffusion matrix of the SDE at time-space (t, x)

        :param t: torch.tensor
            The current time

        :param x: torch.tensor (bs, dim)
            The current value of the process

        :return: torch.tensor
            The diffusion matrix. It's shape depends on diffusion_struct:
                diffusion_struct = 'diag' -> (bs, dim)
                diffusion_struct = 'indep' -> (bs, dim, brown_dim / dim)
                diffusion_struct = 'general' -> (bs, dim, brown_dim)
        """
        pass

    @abstractmethod
    def jumps(self, t, x, jumps):
        """The jump coefficient of the SDE at time-space (t, x)

        :param t: torch.tensor
            The current time

        :param x: torch.tensor (bs, dim)
            The current value of the process

        :param jumps: torch.tensor (bs, dim)
            The current jump sizes

        :return: torch.tensor (bs, dim)
            The jump coefficient of the process at (t, x)
        """
        pass

    @abstractmethod
    def sample_jumps(self, size, device):
        """Method to sample jumps
        :param size: tuple of ints
            The size (shape) of the output of sampled jumps

        :param device: str
            The device on which to sample the jumps

        :return: torch.tensor
            Sampled jumps
        """
        pass

    @abstractmethod
    def jump_mean(self):
        """The expected value of the jump sizes

        :return: torch.tensor
            The mean of the jumps
        """
        pass

    @abstractmethod
    def jump_rate(self):
        """The number of jumps per unit time

        :return: torch.tensor
            The rate of the Poisson process
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

    def jumps(self, t, x, jumps):
        return None

    def sample_jumps(self, size, device):
        return None

    def jump_mean(self):
        return None

    def jump_rate(self):
        return torch.tensor(0)


class Gbm(DiffusionSde):
    """Multi-dimensional correlated geometric Brownian motion (GBM)"""

    def __init__(self, mu, sigma, init_value, dim, corr_matrix=None, method='euler'):
        """
        :param mu: torch.tensor
            The drifts of the process

        :param sigma: torch.tensor
            The volatilities of the process

        :param init_value: torch.tensor
            The initial value of the process

        :param dim: torch.tensor
            The dimension of the GBM

        :param corr_matrix: torch.tensor
            The correlation matrix
        """
        super(Gbm, self).__init__(init_value, dim, dim, 'diag', corr_matrix, method)
        self.mu = mu
        self.sigma = sigma

    def drift(self, t, x):
        return self.mu * x

    def diffusion(self, t, x):
        return self.sigma * x


class LogGbm(Gbm):
    """One-dimensional log GBM - see docs for Gbm class"""
    def __init__(self, mu, sigma, init_value):
        super(LogGbm, self).__init__(mu, sigma, init_value, 1)

    def drift(self, t, x):
        return (self.mu - 0.5 * self.sigma * self.sigma) * torch.ones_like(x)

    def diffusion(self, t, x):
        return self.sigma * torch.ones_like(x)


class DoubleGbm(DiffusionSde):
    """Geometric Brownian motion driven by two independent d-dimensional Brownian motions, used for testing the
    'indep' method for solving"""

    def __init__(self, mu, sigma1, sigma2, init_value, dim, corr_matrix=None, method='euler'):
        """
        :param mu: torch.tensor
            The drifts of the process

        :param sigma1: torch.tensor
            The volatilities for the first Brownian motion

        :param sigma2: torch.tensor
            The volatilities for the second Brownian motion

        :param init_value: torch.tensor
            The initial value of the process

        :param dim: torch.tensor
            The dimension of the GBM

        :param corr_matrix: torch.tensor
            The correlation matrix
        """
        super(DoubleGbm, self).__init__(init_value, dim, dim*2, 'indep', corr_matrix, method)
        self.mu = mu
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def drift(self, t, x):
        return self.mu * x

    def diffusion(self, t, x):
        return torch.stack([self.sigma1 * x, self.sigma2 * x], dim=-1)
    

class Heston(DiffusionSde):
    def __init__(self, r, kappa, theta, xi, rho, init_value):
        assert 2 * kappa * theta > xi**2
        corr_matrix = torch.tensor([[1., rho], [rho, 1.]])
        super(Heston, self).__init__(init_value, 2, 2, 'diag', corr_matrix)
        self.simulation_method = 'heston'
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi

    def drift(self, t, x):
        return torch.stack([self.r * x[:, 0], torch.zeros_like(x[:, 1])], dim=1)

    def diffusion(self, t, x):
        return torch.stack([(torch.sqrt(x[:, 1]) * x[:, 0]), torch.zeros_like(x[:, 1])], dim=1)

    def quadratic_parameters(self, x, h, normals):
        a = - torch.ones_like(x) - self.kappa * h
        b = self.xi * normals
        c = x + self.kappa * self.theta * h - 0.5 * h * self.xi**2
        return a, b, c


class LogNormalJumpsSde(Sde):
    """SDE with jumps that have shifted log-normal distribution"""

    def __init__(self, rate, alpha, gamma, init_value, dim, brown_dim, diffusion_struct, corr_matrix=None, method='euler'):
        """
        :param rate: float, torch.tensor
            The rate of the Poisson process

        :param alpha: float
            The mean of the jumps

        :param gamma: float
            The standard deviation of the jumps

        :param init_value: torch.tensor
            The initial value of the process

        :param dim: int
            The dimension of the SDE

        :param corr_matrix: torch.tensor
            The correlation matrix
        """
        super(LogNormalJumpsSde, self).__init__(init_value, dim, brown_dim, diffusion_struct, corr_matrix, method)
        if torch.is_tensor(rate):
            self.rate = rate
        else:
            self.rate = torch.tensor(rate)
        self.alpha = alpha
        self.gamma = gamma

    @abstractmethod
    def drift(self, t, x):
        pass

    @abstractmethod
    def diffusion(self, t, x):
        pass

    @abstractmethod
    def jumps(self, t, x, jumps):
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
        :param mu: float
            The drift of the process

        :param sigma: float
            The volatility of the process

        :param rate: float
            The rate of the Poisson process

        :param alpha: float
            The mean of the jumps

        :param gamma: float
            The standard deviation of the jumps

        :param init_value: torch.tensor
            The initial value of the process

        :param dim: int
            The dimension of the SDE

        :param corr_matrix: torch.tensor
            The correlation matrix
        """
        self.mu = mu
        self.sigma = sigma
        super(Merton, self).__init__(rate, alpha, gamma, init_value, dim, dim, 'diag', corr_matrix, method)

    def drift(self, t, x):
        return (self.mu - self.rate * self.jump_mean()) * x

    def diffusion(self, t, x):
        return self.sigma * x

    def jumps(self, t, x, jumps):
        return x * jumps


class AsianWrapper(Sde):
    """Wraps an existing 1-D SDE object and computes the integral of the process over time - for Asian options"""

    def __init__(self, base_sde):
        super().__init__(torch.cat([base_sde.init_value, torch.tensor([0.])]),
                         2,
                         2,
                         'diag',
                         None,
                         base_sde.simulation_method)
        self.base_sde = base_sde

    def drift(self, t, x):
        return torch.stack([self.base_sde.drift(t, x[:, 0]), x[:, 0]], dim=1)

    def diffusion(self, t, x):
        return torch.stack([self.base_sde.diffusion(t, x[:, 0]), torch.zeros_like(x[:, 0])], dim=1)

    def jumps(self, t, x, jumps):
        return torch.stack([self.base_sde.jumps(t, x[:, 0], jumps[:, 0]), torch.zeros_like(x[:, 0])], dim=1)

    def sample_jumps(self, size, device):
        return self.base_sde.sample_jumps(size, device)

    def jump_mean(self):
        return self.base_sde.jump_mean()

    def jump_rate(self):
        return self.base_sde.jump_rate()
