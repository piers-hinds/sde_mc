from regression import fit_basis
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from functools import partial


class SdeApproximator(ABC):
    """Abstract base class for an approximator to the solution of the PDE associated with the SDE"""

    def __init__(self, time_points):
        """
        :param time_points: torch.tensor, the time points at which the data is observed
        """
        self.time_points = time_points

    @abstractmethod
    def fit(self, paths, payoffs):
        pass

    @abstractmethod
    def __call__(self, time_idx, t, x):
        pass


class LinearApproximator(SdeApproximator):
    """Abstract class for the approximate solutions to the SDEs using linear regression"""

    def __init__(self, basis, time_points):
        """
        :param basis: list of functions, the basis functions to fit to the data
        """
        super(LinearApproximator, self).__init__(time_points=time_points)
        self.basis = basis
        self.coefs = None

    def fit(self, paths, payoffs):
        """Fits the basis functions to the data (paths, payoffs)
        :param paths: torch.tensor, the paths as outputted from an SdeSolver
        :param payoffs: torch.tensor, the payoffs as outputted from an SdeSolver
        :return: None
        """
        self.coefs = torch.empty((paths.shape[1] - 1, len(self.basis)))
        for i in range(1, paths.shape[1]):
            current_basis = [partial(b, t=self.time_points[i-1]) for b in self.basis]
            self.coefs[i-1] = fit_basis(paths[:, i].squeeze(-1), payoffs, current_basis)

    def derivative(self, time_idx, x):
        """Computes the derivative at (time_idx, x) using automatic differentiation
        :param time_idx: int, the index of the time points at which to evaluate the basis functions
        :param x: torch.tensor, the value of the process
        :return: torch.tensor, the approximate solution
        """
        basis_sum = 0
        for i, b in enumerate(self.basis):
            x.requires_grad = True
            y = b(x, self.time_points[time_idx])
            y.backward(torch.ones_like(x))
            x.requires_grad = False
            grad = x.grad
            x.grad = None
            basis_sum += self.coefs[time_idx, i] * grad
        return basis_sum

    @abstractmethod
    def __call__(self, time_idx, t, x):
        """Override with your code here - depends on the PDE
        :param time_idx: int, the index of the time points
        :param t: torch.tensor
        :param x: torch.tensor
        :return: the approximation for F * Y (i.e. the diffusion term of the control variate process Z)
        """
        pass


class GbmLinear(LinearApproximator):
    """Approximator for GBM"""

    def __init__(self, basis, time_points, mu, sigma):
        super(GbmLinear, self).__init__(basis, time_points)
        self.mu = mu
        self.sigma = sigma

    def __call__(self, time_idx, t, x):
        return torch.exp(-self.mu * t) * -self.sigma * x * self.derivative(time_idx, x)


class Mlp(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        assert len(layer_sizes) > 0, "At least one hidden layer required."
        super(Mlp, self).__init__()
        self.num_layers = len(layer_sizes)

        layers = [nn.Linear(input_size, layer_sizes[0]), nn.BatchNorm1d(layer_sizes[0]), nn.ReLU()]
        for i in range(self.num_layers-1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.BatchNorm1d(layer_sizes[i+1]), nn.ReLU()]
        layers += [nn.Linear(layer_sizes[self.num_layers-1], output_size)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NetApproximator(SdeApproximator):
    """Abstract class for approximate solutions using a feed-forward network"""

    def __init__(self, time_points, arch):
        super(NetApproximator, self).__init__(time_points)
        self.arch = arch

    def fit(self, paths, payoffs):
        pass

    @abstractmethod
    def __call__(self, time_idx, t, x):
        pass

