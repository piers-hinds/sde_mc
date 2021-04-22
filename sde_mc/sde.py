from abc import ABC, abstractmethod
import torch


class Sde(ABC):
    """Abstract base class for SDEs"""

    def __init__(self, init_value):
        """
        :param init_value: torch.tensor, the (possibly multi-dimensional) initial value of the SDE
        """
        self.init_value = init_value

    @abstractmethod
    def drift(self, t, x):
        """The drift function associated with the SDE
        :param t: torch.tensor, the current time
        :param x: torch.tensor, the current value of the SDE
        :return: torch.tensor, the drift at the current time and value
        """
        pass

    @abstractmethod
    def diffusion(self, t, x):
        """The diffusion function associated with the SDE
        :param t: torch.tensor, the current time
        :param x: torch.tensor, the current value of the SDE
        :return: torch.tensor, the diffusion at the current time and value
        """
        pass


class Gbm(Sde):
    """A class for geometric Brownian motion: dX = "mu"X dt + "sigma"X dW"""

    def __init__(self, mu, sigma, init_value):
        """
        :param mu: float, the drift
        :param sigma: float, the volatility
        :param init_value: torch.tensor, the (possibly multi-dimensional) initial value of the SDE
        """
        super(Gbm, self).__init__(init_value)
        self.mu = mu
        self.sigma = sigma

    def drift(self, t, x):
        return self.mu * x

    def diffusion(self, t, x):
        return self.sigma * x


class SdeSolver:
    """A class for solving SDEs"""

    def __init__(self, sde, time, num_steps, dimension, device='cpu', seed=1):
        """
        :param sde: Sde, the underlying SDE
        :param time: float, the final time point to simulate the SDE up to
        :param num_steps: int, the number of time steps used in the discretization
        :param dimension: int, the dimension of the process
        :param device: string, the device to do the computation on (see torch docs)
        :param seed: int, seed for torch
        """
        assert len(sde.init_value) == dimension, "The dimension should match the dimension of the SDE's initial value"
        self.sde = sde
        self.time = time
        self.num_steps = num_steps
        self.dimension = dimension
        self.device = device

        self.h = torch.tensor(self.time / self.num_steps, device=self.device)
        torch.manual_seed(seed)

    def euler(self, bs=1):
        """Implements the Euler scheme for SDEs
        :param bs: int, the batch size (the number of paths computed at one time)
        :return: torch.tensor, the generated paths across batches, time points and dimensions
        """
        paths = torch.empty(size=(bs, self.num_steps + 1, self.dimension), device=self.device)
        paths[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        rvs = torch.randn(size=(bs, self.num_steps, self.dimension), device=self.device) * torch.sqrt(self.h)

        t = torch.tensor(0.0, device=self.device)
        for i in range(self.num_steps):
            paths[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * self.h + \
                     self.sde.diffusion(t, paths[:, i]) * rvs[:, i]
            t += self.h
        return paths
