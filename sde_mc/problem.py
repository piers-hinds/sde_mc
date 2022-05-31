from abc import ABC
import torch
from .sde import Gbm, Heston
from .solvers import EulerSolver
from .options import EuroCall, ConstantShortRate


class Problem(ABC):
    def __init__(self, solver, discounter, payoff):
        self.solver = solver
        self.discounter = discounter
        self.payoff = payoff

    def dim(self):
        return self.solver.sde.dim


class BlackScholesEuroCall(Problem):
    def __init__(self, r, sigma, spot, strike, maturity, steps, device):
        gbm = Gbm(r, sigma, torch.tensor([spot]), 1)
        solver = EulerSolver(gbm, maturity, steps, device)
        csr = ConstantShortRate(r)
        option = EuroCall(strike)
        super().__init__(solver, csr, option)


class HestonEuroCall(Problem):
    def __init__(self, r, kappa, theta, xi, rho, spot, v0, strike, maturity, steps, device):
        heston = Heston(r, kappa, theta, xi, rho, torch.tensor([spot, v0]))
        solver = EulerSolver(heston, maturity, steps, device)
        csr = ConstantShortRate(r)
        option = EuroCall(strike)
        super().__init__(solver, csr, option)
