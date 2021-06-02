from sde_mc import *
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial
from abc import ABC, abstractmethod

# steps = 4
# trials = 3
#
# x0 = torch.tensor([1., 0.25])
# heston = Heston(r=0.02, kappa=0.2, theta=0.3, xi=0.1, rho=-0.2, init_value=x0)
# solver = HestonSolver(heston, 3, steps)
# paths = solver.euler(bs=trials)
# print(paths)

steps = 100
trials = 1000
bs = None

gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0] * 2), dim=2)
solver = SdeSolver(sde=gbm, time=3, num_steps=steps)

mc_stats = mc_simple(trials, solver, Rainbow(strike=1.), np.exp(-0.06), bs=bs)
mc_stats.print()

mlp = Mlp(gbm.dim + 1, [30, 30], gbm.dim)
stats = mc_min_variance(trials=(1000, 1000), solver=solver, model=mlp, payoff=Rainbow(strike=1.), discounter=ConstantShortRate(r=0.02),
                        bs=(100, 1000), step_factor=30, epochs=4)
stats.print()























