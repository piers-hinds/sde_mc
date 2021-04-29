from abc import ABC, abstractmethod
import torch


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
        :return: torch.tensor (bs, dim), the drift of the process at (t, x)
        """
        pass

    @abstractmethod
    def diffusion(self, t, x):
        """YOUR CODE HERE
        :param t: torch.tensor, the current time
        :param x: torch.tensor (bs, dim), the current value of the process
        :return: torch.tensor (bs, dim), the diffusion of the process at (t, x)
        """
        pass


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


class SdeSolver:
    """A class for solving SDEs"""

    def __init__(self, sde, time, num_steps, device='cpu', seed=1):
        """
        :param sde: Sde, the SDE to solve
        :param time: float, the time to solve up to
        :param num_steps: int, the number of steps in the discretization
        :param device: string, the device to do the computations on
        :param seed: int, seed for torch
        """
        self.sde = sde
        self.time = time
        self.num_steps = num_steps
        self.device = device

        self.h = torch.tensor(self.time / self.num_steps, device=self.device)
        if len(self.sde.corr_matrix) > 1:
            self.L = torch.linalg.cholesky(self.sde.corr_matrix.to(device))
        else:
            self.L = torch.tensor([[1.]], device=device)
        torch.manual_seed(seed)

    def euler(self, bs=1):
        """Implements the Euler method for solving SDEs
        :param bs: int, the batch size (the number of paths simulated simultaneously)
        :return: torch.tensor, the paths simulated across (bs, steps, dimensions)
        """
        assert bs >= 1, "Batch size must at least one"
        bs = int(bs)

        paths = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)
        paths[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

        rvs = torch.randn(size=(bs, self.num_steps, self.sde.noise_dim, 1), device=self.device) * torch.sqrt(self.h)
        rvs = torch.matmul(self.L, rvs)
        t = torch.tensor(0.0, device=self.device)
        for i in range(self.num_steps):
            paths[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * self.h + \
                              torch.matmul(self.sde.diffusion(t, paths[:, i]), rvs[:, i]).squeeze(-1)
            t += self.h
        return paths
