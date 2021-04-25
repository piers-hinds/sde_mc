from options import aon_payoff, aon_true
from sde import Gbm, SdeSolver
from mc import mc_simple
from regression import *
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial

steps = 10
trials = 1000

gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0, 1.0]))
solver = SdeSolver(sde=gbm, time=3, num_steps=steps, dimension=2)

mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=aon_payoff, discount=np.exp(-0.06), bs=None)
mc_stats.print()


"""
paths = mc_stats.paths
payoffs = mc_stats.payoffs
idx = 500
x = paths[:, idx].squeeze(-1)
print(x)
t = torch.tensor([idx * 3 / steps])
aon_basis = [partial(basis_1, t=t), partial(basis_2, t=t), partial(basis_3, t=t)]
test = fit_basis(x, payoffs, aon_basis)
plt.scatter(x, payoffs)
plot_basis(test, aon_basis, x.min(), x.max())
plt.show()
"""