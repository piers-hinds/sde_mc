from .sde import Sde
from .regression import fit_basis
import torch
from abc import ABC, abstractmethod
from functools import partial


class ControlVariateApproximator(ABC):
    """Abstract class for the approximate solutions to the SDEs"""

    def __init__(self, basis, ts):
        """
        :param basis: list of functions, the basis functions to fit to the data
        :param ts: torch.tensor, the time points at which the data is observed
        """
        self.basis = basis
        self.ts = ts
        self.coefs = None

    def fit(self, paths, payoffs):
        """Fits the basis functions to the data (paths, payoffs)
        :param paths: torch.tensor, the paths as outputted from an SdeSolver
        :param payoffs: torch.tensor, the payoffs as outputted from an SdeSolver
        :return: None
        """
        self.coefs = torch.empty((paths.shape[1] - 1, len(self.basis)))
        for i in range(1, paths.shape[1]):
            current_basis = [partial(b, t=self.ts[i-1]) for b in self.basis]
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
            y = b(x, self.ts[time_idx])
            y.backward(torch.ones_like(x))
            x.requires_grad = False
            grad = x.grad
            x.grad = None
            basis_sum += self.coefs[time_idx, i] * grad
        return basis_sum

    @abstractmethod
    def __call__(self, time_idx, t, x):
        """Override with your code here - depends on the Sde
        :param time_idx: int, the index of the time points
        :param t: torch.tensor
        :param x: torch.tensor
        :return: the approximation for F * Y (i.e. the diffusion term of the control variate process Z)
        """
        pass


class GbmApproximator(ControlVariateApproximator):
    """Approximator for GBM"""

    def __init__(self, basis, ts, mu, sigma):
        super(GbmApproximator, self).__init__(basis, ts)
        self.mu = mu
        self.sigma = sigma

    def __call__(self, time_idx, t, x):
        return torch.exp(-self.mu * t) * -self.sigma * x * self.derivative(time_idx, x)


class SdeControlVariate(Sde):
    """An Sde class which adds a control variate to an existing SDE"""

    def __init__(self, base_sde, control_variate, time_points):
        """
        :param base_sde: Sde, the original SDE to add a control variate to
        :param control_variate: ControlVariateApproximator, the approximation of F * Y
        :param time_points: torch.tensor, the time points at which to update the control variate
        """
        super(SdeControlVariate, self).__init__(torch.cat([base_sde.init_value, torch.tensor([0.])]))
        self.base_sde = base_sde
        self.base_dim = base_sde.dim
        self.cv = control_variate
        self.time_points = time_points

        self.idx = 0
        self.F = 0

    def drift(self, t, x):
        return torch.cat([self.base_sde.drift(t, x[:, :self.base_dim]), torch.zeros_like(x[:, self.base_dim]).unsqueeze(1)], dim=-1)

    def diffusion(self, t, x):
        if not t:
            self.reset_control(x)
        if t >= self.time_points[self.idx]:
            self.idx += 1
            self.update_control(t, x)

        return torch.cat([self.base_sde.diffusion(t, x[:, :self.base_dim]), self.F], dim=-1)

    def update_control(self, t, x):
        """Updates the control variate"""
        self.F = self.cv(self.idx, t, x[:, :self.base_dim])

    def reset_control(self, x):
        """Resets the control variate (e.g. when restarting)"""
        self.F = torch.zeros_like(x[:, :self.base_dim])
        self.idx = 0
