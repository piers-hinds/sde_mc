from abc import ABC, abstractmethod
import torch
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

        if len(self.sde.corr_matrix) > 1:
            self.lower_cholesky = torch.linalg.cholesky(self.sde.corr_matrix.to(device))
        else:
            self.lower_cholesky = torch.tensor([[1.]], device=device)
        torch.manual_seed(seed)

    def euler(self, bs=1, return_normals=False):
        """Implements the Euler method for solving SDEs
        :param bs: int, the batch size (the number of paths simulated simultaneously)
        :param return_normals: bool, if True returns the normal random variables used
        :return: torch.tensor, the paths simulated across (bs, steps, dimensions)
        """
        assert bs >= 1, "Batch size must at least one"
        bs = int(bs)

        h = torch.tensor(self.time / self.num_steps, device=self.device)

        paths = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)
        paths[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

        normals = torch.randn(size=(bs, self.num_steps, self.sde.noise_dim, 1), device=self.device) * torch.sqrt(h)
        corr_normals = torch.matmul(self.lower_cholesky, normals)

        t = torch.tensor(0.0, device=self.device)
        for i in range(self.num_steps):
            paths[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * h + \
                              torch.matmul(self.sde.diffusion(t, paths[:, i]), corr_normals[:, i]).squeeze(-1)
            t += h

        if return_normals:
            return paths, corr_normals
        else:
            return paths, None


class HestonSolver(SdeSolver):
    """A solver class specifically designed for the Heston model, where the volatility should be simulated using a
    different scheme (fully-implicit) to the asset price (explicit) to preserve positivity of the volatility."""

    def euler(self, bs=1, return_normals=False):
        assert bs >= 1, "Batch size must at least one"
        bs = int(bs)

        paths = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)
        paths[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

        normals = torch.randn(size=(bs, self.num_steps, self.sde.noise_dim, 1), device=self.device) * torch.sqrt(self.h)
        corr_normals = torch.matmul(self.lower_cholesky, normals).squeeze(-1)

        t = torch.tensor(0.0, device=self.device)

        for i in range(self.num_steps):
            # Calculate Xk as normal
            paths[:, i+1, 0] = paths[:, i, 0] + self.sde.r * paths[:, i, 0] * self.h +\
                               paths[:, i, 0] * torch.sqrt(paths[:, i, 1]) * corr_normals[:, i, 0]
            # Solve for sqrt V_k+1
            # A, B, C =
            # paths = quad formula ^2
            a = (1 - self.h * self.sde.kappa)
            c = (paths[:, i, 1] + self.h * self.sde.kappa * self.sde.theta - self.h * self.sde.xi * self.sde.xi * 0.5)
            b = (self.sde.xi * corr_normals[:, i, 1])

            paths[:, i+1, 1] = ((- b + torch.sqrt(b * b - 4 * a * c)) / (2 * a)) ** 2

        return paths


class StackSde(Sde):
    """A class to stack multiple SDEs into one multidimensional SDEs"""
    def __init__(self, sdes):
        dim = 0
        for sde in sdes:
            dim += sde.dim
        super(StackSde, self).__init__()

    def drift(self, t, x):
        pass

    def diffusion(self, t, x):
        pass


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


class JumpSolver(SdeSolver):
    def euler_jumps(self, rate=1, m=0, v=0.3, bs=1, return_normals=False):
        assert bs >= 1, "Batch size must at least one"
        bs = int(bs)

        h = torch.tensor(self.time / self.num_steps, device=self.device)

        paths = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)
        paths[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

        normals = torch.randn(size=(bs, self.num_steps, self.sde.noise_dim, 1), device=self.device) * torch.sqrt(h)
        corr_normals = torch.matmul(self.lower_cholesky, normals)

        rates = torch.ones(size=(bs, self.num_steps, self.sde.dim), device=self.device) * (rate * h)
        poissons = torch.poisson(rates)
        max_jumps = (torch.randn(size=(bs, self.num_steps, self.sde.dim), device=self.device) * v).exp() - 1
        jumps = (max_jumps * torch.gt(poissons, 0))

        t = torch.tensor(0.0, device=self.device)
        for i in range(self.num_steps):
            paths[:, i + 1] = (paths[:, i] + self.sde.drift(t, paths[:, i]) * h + \
                              torch.matmul(self.sde.diffusion(t, paths[:, i]), corr_normals[:, i]).squeeze(-1)) * \
                              (jumps[:, i] + 1)
            t += h

        if return_normals:
            return paths, corr_normals
        else:
            return paths, None