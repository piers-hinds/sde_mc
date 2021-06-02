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
trials = 5000
bs = 100

gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
solver = SdeSolver(gbm, 3, steps)

mc_stats = mc_simple(trials, solver, BinaryAoN(1.), np.exp(-0.06), bs=None)
mc_stats.print()























