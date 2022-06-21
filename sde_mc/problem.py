from abc import ABC
import torch
from .sde import Gbm, Heston, Merton
from .solvers import EulerSolver, JumpEulerSolver
from .options import EuroCall, ConstantShortRate


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


class HestonEuroCall(Problem):
    def __init__(self, r, kappa, theta, xi, rho, spot, v0, strike, maturity, steps, device):
        heston = Heston(r, kappa, theta, xi, rho, torch.tensor([spot, v0]))
        solver = EulerSolver(heston, maturity, steps, device)
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
        return MertonEuroCall(0.02, 0.3, 2, -0.05, 0.3, 1, 1, 3, steps, device)

