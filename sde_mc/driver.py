from sde_mc.options import aon_payoff, aon_true
from sde_mc.sde import Gbm, SdeSolver, Sde, Heston
from sde_mc.mc import mc_simple, mc_control_variate
from sde_mc.regression import *
from sde_mc.vreduction import GbmApproximator, SdeControlVariate
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial
from abc import ABC, abstractmethod


def heston_call(log_spot, strike):
    spot = torch.exp(log_spot[:, 0])
    return torch.where(spot > strike, spot-strike, torch.tensor(0., dtype=spot.dtype, device=spot.device))


def bs_call(spot, strike):
    return torch.where(spot > strike, spot-strike, torch.tensor(0., dtype=spot.dtype, device=spot.device))


test = torch.tensor([[0.7, 3.], [1.9, 0.5]])
print(heston_call(test, 1))

x0 = torch.tensor([np.log(1), np.log(0.2**2)])
heston = Heston(r=0.02, kappa=0.2, theta=0.2**2, xi=0.1, rho=-0.2, init_value=x0)
x = torch.tensor([[1., 0.1]])
my_payoff = partial(heston_call, strike=1)
solver = SdeSolver(sde=heston, time=3, num_steps=1000)

# paths = solver.euler()
# print(paths)
mc_stats = mc_simple(10000, sde_solver=solver, payoff=my_payoff, discount=np.exp(-0.06))
mc_stats.print()

x0 = torch.tensor([1.])
gbm = Gbm(mu=0.02, sigma=0.196, init_value=x0, dim=1)
gbm_solver = SdeSolver(sde=gbm, time=3, num_steps=1000)
mc_stats_gbm = mc_simple(10000, sde_solver=gbm_solver, payoff=partial(bs_call, strike=1), discount=np.exp(-0.06))
mc_stats_gbm.print()

# Example which shows control variates 4x faster
# steps = 3000
# trials = 70000
# my_payoff = partial(aon_payoff, strike=1)
#
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
#
# mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=my_payoff, discount=np.exp(-0.06))
# mc_stats.print()
#
# steps = 600
# ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
# gbm_approx = GbmApproximator(basis=[basis_1, basis_2, basis_3], ts=ts, mu=gbm.mu, sigma=gbm.sigma)
# new_cv_stats = mc_control_variate(num_trials=(2500, 5000), simple_solver=solver, approximator=gbm_approx,
#                                   payoff=my_payoff, discount=np.exp(-0.06))
# new_cv_stats.print()
