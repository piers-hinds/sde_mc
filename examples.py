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

r = torch.tensor([0.02, 0.02])
kappa = torch.tensor([0.4, 0.4])
theta = torch.tensor([0.2**2, 0.2**2])
xi = torch.tensor([0.1, 0.1])
rho = torch.tensor([-0.3, -0.3])
x0 = torch.tensor([0., np.log(0.2**2), 0., np.log(0.2**2)])

heston = Heston(0.02, 0.4, 0.2**2, 0.1, -0.3, torch.tensor([0., np.log(0.2**2)]))
multi_heston = MultiHeston(r=r, kappa=kappa, theta=theta, xi=xi, rho=rho, init_value=x0, dim=2)

solver = SdeSolver(multi_heston, 3, 100)
mc_stats = mc_simple(1000, solver, HestonRainbow(strike=1, log=True), discount=np.exp(-0.06))
mc_stats.print()























