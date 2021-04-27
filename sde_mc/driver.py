from options import aon_payoff, aon_true
from sde import Gbm, SdeSolver
from mc import mc_simple
from regression import *
from vreduction import GbmApproximator, SdeControlVariate
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial


def aon_payoff_cv(spot, strike):
    return aon_payoff(spot[:, 0], strike) * np.exp(-0.06) + spot[:, 1]


steps = 600
trials = 10000

gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]))
solver = SdeSolver(sde=gbm, time=3, num_steps=steps, dimension=1)

mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=aon_payoff, discount=np.exp(-0.06), bs=None, shared_noise=True)
mc_stats.print()

ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
gbm_approx = GbmApproximator(basis=[basis_1, basis_2, basis_3], ts=ts, mu=gbm.mu, sigma=gbm.sigma)
gbm_approx.fit(paths=mc_stats.paths, payoffs=mc_stats.payoffs)

gbm_cv = SdeControlVariate(base_sde=gbm, control_variate=gbm_approx, time_points=ts)
cv_solver = SdeSolver(sde=gbm_cv, time=3, num_steps=steps, dimension=2)
mc_stats_cv = mc_simple(num_trials=10000, sde_solver=cv_solver, payoff=aon_payoff_cv, shared_noise=True)
mc_stats_cv.print()


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