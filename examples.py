from sde_mc import *
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial
from abc import ABC, abstractmethod


S = 1.
K = 1
T = 3
r = 0.02
sigma = 0.2
m = 0
v = 0.3
lam = 2
trials = 100000

call = EuroCall(strike=K)

true_price = merton_jump_call(S, K, T, r, sigma, m, v, lam)
print(true_price)

my_price = merton_call(S, K, T, r, sigma, m, v, 0, lam)
print(my_price)

# gbm = Gbm(mu=r-lam*(np.exp(v**2*0.5) - 1), sigma=sigma, init_value=torch.tensor([S]), dim=1)
# solver = JumpSolver(sde=gbm, time=T, num_steps=1000, seed=3)
# paths, jumps = solver.euler(rate=lam, v=v, bs=trials)
#
# payoffs = call(paths[:, solver.num_steps])
# print(payoffs.mean() * np.exp(-r*T), 2*payoffs.std()/np.sqrt(trials))





















