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

























