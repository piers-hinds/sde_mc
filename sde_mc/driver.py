from options import aon_payoff, aon_true
from sde import Gbm, SdeSolver
from mc import mc_simple
import torch
import numpy as np
import time

steps = 800
trials = 10000

gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]))
solver = SdeSolver(sde=gbm, time=3, num_steps=steps, dimension=1)

mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=aon_payoff, discount=np.exp(-0.06))
mc_stats.print()

