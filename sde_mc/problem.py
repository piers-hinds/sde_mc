from abc import ABC
import torch
from .sde import Gbm, Heston, Merton
from .solvers import EulerSolver, JumpEulerSolver, HestonSolver
from .options import EuroCall, ConstantShortRate, Rainbow, BestOf
from .levy import LevySde, ExpExampleLevy
from .helpers import get_corr_matrix


class Problem(ABC):
    def __init__(self, solver, discounter, payoff):
        self.solver = solver
        self.discounter = discounter
        self.payoff = payoff

    def dim(self):
        return self.solver.sde.dim

    def set_steps(self, steps):
        self.solver.num_steps = steps


class BlackScholesEuroCall(Problem):
    def __init__(self, r, sigma, spot, strike, maturity, steps, device):
        gbm = Gbm(r, sigma, torch.tensor([spot]), 1)
        solver = EulerSolver(gbm, maturity, steps, device)
        csr = ConstantShortRate(r)
        option = EuroCall(strike)
        super().__init__(solver, csr, option)

    @classmethod
    def default_params(cls, steps, device):
        return BlackScholesEuroCall(0.02, 0.3, 1, 1, 3, steps, device)


class BlackScholesRainbow(Problem):
    def __init__(self, r, sigma, spot, strike, maturity, dim, corr_matrix, steps, device):
        init_value = torch.ones(dim) * spot
        gbm = Gbm(r, sigma, init_value, dim, corr_matrix)
        solver = EulerSolver(gbm, maturity, steps, device)
        csr = ConstantShortRate(r)
        option = Rainbow(strike)
        super().__init__(solver, csr, option)

    @classmethod
    def default_params(cls, steps, device):
        corr_matrix = get_corr_matrix([0.7, 0.2, -0.3])
        return BlackScholesRainbow(0.02, 0.3, 1, 1, 3, 3, corr_matrix, steps, device)


class HestonEuroCall(Problem):
    def __init__(self, r, kappa, theta, xi, rho, spot, v0, strike, maturity, steps, device):
        heston = Heston(r, kappa, theta, xi, rho, torch.tensor([spot, v0]))
        solver = HestonSolver(heston, maturity, steps, device)
        csr = ConstantShortRate(r)
        option = EuroCall(strike)
        super().__init__(solver, csr, option)

    @classmethod
    def default_params(cls, steps, device):
        return HestonEuroCall(0.02, 0.25, 0.5, 0.3, -0.3, 1, 0.15, 1, 3, steps, device)


class MertonEuroCall(Problem):
    def __init__(self, mu, sigma, rate, alpha, gamma, spot, strike, maturity, steps, device):
        merton = Merton(mu, sigma, rate, alpha, gamma, torch.tensor([spot]), 1)
        solver = JumpEulerSolver(merton, maturity, steps, device)
        csr = ConstantShortRate(mu)
        option = EuroCall(strike)
        super().__init__(solver, csr, option)

    @classmethod
    def default_params(cls, steps, device):
        return MertonEuroCall(0.02, 0.2, 1, -0.05, 0.3, 1, 1, 3, steps, device)


class LevyRainbow(Problem):
    def __init__(self, c_minus, c_plus, alpha, mu, r, sigma, f, epsilon, dim, spot, strike, maturity, steps, device):
        if not torch.is_tensor(spot):
            spot = torch.ones(dim) * spot
        exptest = ExpExampleLevy(c_minus, c_plus, alpha, mu, r, sigma, f, epsilon, dim)
        expsde = LevySde(exptest, spot, device=device)
        solver = JumpEulerSolver(expsde, maturity, steps, device=device)
        csr = ConstantShortRate(r)
        option = Rainbow(strike)
        super().__init__(solver, csr, option)

    @classmethod
    def default_params(cls, steps, device):
        return LevyRainbow(1, 1, 0.5, 2, 0.02, 0.3, 0.2, 0.001, 2, 1, 1, 3, steps, device)


class LevyCall(Problem):
    def __init__(self, c_minus, c_plus, alpha, mu, r, sigma, f, epsilon, spot, strike, maturity, steps, device):

        exptest = ExpExampleLevy(c_minus, c_plus, alpha, mu, r, sigma, f, epsilon, 1)
        expsde = LevySde(exptest, torch.tensor([spot]), device=device)
        solver = JumpEulerSolver(expsde, maturity, steps, device=device)
        csr = ConstantShortRate(r)
        option = EuroCall(strike)
        super().__init__(solver, csr, option)

    @classmethod
    def default_params(cls, steps, device):
        return LevyCall(1, 1, 0.5, 2, 0.02, 0.3, 0.2, 0.001, 1, 1, 3, steps, device)


class LevyBestOf(Problem):
    def __init__(self, c_minus, c_plus, alpha, mu, r, sigma, f, epsilon, dim, spot, strike, maturity, steps, device):
        if not torch.is_tensor(spot):
            spot = torch.ones(dim) * spot
        exptest = ExpExampleLevy(c_minus, c_plus, alpha, mu, r, sigma, f, epsilon, dim)
        expsde = LevySde(exptest, spot, device=device)
        solver = JumpEulerSolver(expsde, maturity, steps, device=device)
        csr = ConstantShortRate(r)
        option = BestOf(strike)
        super().__init__(solver, csr, option)

    @classmethod
    def default_params(cls, steps, device):
        return LevyBestOf(1, 1, 0.2, 2, 0.02, 0.3, 0.2, 0.001, 4, 1, 1, 3, steps, device)