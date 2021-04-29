from sde_mc.options import aon_payoff, aon_true
from sde_mc.sde import Gbm, SdeSolver
from sde_mc.mc import mc_simple, mc_control_variate
from sde_mc.regression import *
from sde_mc.vreduction import GbmApproximator, SdeControlVariate
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial
from abc import ABC, abstractmethod


def exchange_payoff(spot):
    payoffs = spot[:, 1] - spot[:, 0]
    return torch.where(payoffs > 0, payoffs, torch.tensor(0., dtype=spot.dtype, device=spot.device))


x = torch.tensor([1., 1.])
my_payoff = partial(aon_payoff, strike=1)
mu = 0.02
sigma = 0.2
cmat = torch.tensor([[1., -0.5], [-0.5, 1.]])
test = Gbm(mu, sigma, x, dim=2, corr_matrix=cmat)
solver = SdeSolver(sde=test, time=3, num_steps=1000)
stats = mc_simple(20000, sde_solver=solver, payoff=exchange_payoff, discount=np.exp(-0.06))
stats.print()


# Example which shows control variates 3x faster
# steps = 3000
# trials = 70000
# my_payoff = partial(aon_payoff, strike=1)
#
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]))
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps, dimension=1)
#
# mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=my_payoff, discount=np.exp(-0.06), shared_noise=True)
# mc_stats.print()
#
# steps = 600
# ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]))
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps, dimension=1)
# gbm_approx = GbmApproximator(basis=[basis_1, basis_2, basis_3], ts=ts, mu=gbm.mu, sigma=gbm.sigma)
# new_cv_stats = mc_control_variate(num_trials=(500, 5000), simple_solver=solver, approximator=gbm_approx,
#                                   payoff=my_payoff, discount=np.exp(-0.06))
# new_cv_stats.print()
