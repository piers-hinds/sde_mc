from sde_mc import *
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial
from abc import ABC, abstractmethod


my_net = Mlp(1, [5, 10, 4], 1)
for child_name, child in my_net.net.named_children():
    print(child_name, child)

setattr(my_net.net, str(9), nn.ReLU())
for child_name, child in my_net.net.named_children():
    print(child_name, child)

# steps = 600
# trials = 100
#
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
#
# mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=BinaryAoN(strike=1.), discount=np.exp(-0.06))
# mc_stats.print()
# paths = mc_stats.paths
# payoffs = mc_stats.payoffs
#
# ts = torch.tensor([t * 3 / steps for t in range(1, steps + 1)])
#
# net_approx = GbmNet(time_points=ts, layer_sizes=[10], mu=0.02, sigma=0.2, final_activation=nn.ReLU(), epochs=3)
# net_approx.fit(paths, payoffs)
#
# lin_approx = GbmLinear(basis=[basis_1, basis_2, basis_3],  time_points=ts,
#                        mu=0.02, sigma=0.2)
# lin_approx.fit(paths, payoffs)
# x = torch.linspace(0.5, 2, 100).unsqueeze(-1)
# t = torch.tensor(2.)
# inputs = torch.cat([t.unsqueeze(-1).repeat(len(x)).unsqueeze(-1), x], dim=-1)
# plt.scatter(paths[:, 400], payoffs)
# plt.plot(x, net_approx.mlp(inputs).detach())
# plt.show()

# grads_net = net_approx(0, t, x)
# grads_lin = lin_approx(400, t, x)
# plt.plot(x, grads_net)
# plt.plot(x, grads_lin, color='green')
# plt.show()



# Example which shows control variates 3-4x faster with Net approximation
# steps = 3000
# trials = 10000
#
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
# paths = solver.euler(2)
#
# mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=BinaryAoN(strike=1.), discount=np.exp(-0.06))
# mc_stats.print()
#
# steps = 100
# ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
# net_approx = GbmNet(layer_sizes=[10, 10], time_points=ts, mu=gbm.mu, sigma=gbm.sigma)
# new_cv_stats = mc_control_variate(num_trials=(200, 5000), simple_solver=solver, approximator=net_approx,
#                                   payoff=BinaryAoN(strike=1.), discount=np.exp(-0.06))
# new_cv_stats.print()







#
#
# idx = 1
# t = idx*3/steps
# basis = [partial(basis_1, t=t), partial(basis_2, t=t), partial(basis_3, t=t)]
# print(payoffs)
# print(paths[:, idx].squeeze(-1))
# print(fit_basis(paths[:, 1].squeeze(-1), payoffs, basis))


# # Heston example:
# x0 = torch.tensor([np.log(1), np.log(0.2**2)])
# heston = Heston(r=0.02, kappa=0.2, theta=0.2**2, xi=0.1, rho=-0.2, init_value=x0)
# solver = SdeSolver(sde=heston, time=3, num_steps=1000)
# mc_stats = mc_simple(10000, sde_solver=solver, payoff=EuroCall(1., log=True), discount=np.exp(-0.06))
# mc_stats.print()


# # Example which shows control variates 3-4x faster with linear approximation
# steps = 3000
# trials = 70000
#
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
# paths = solver.euler(2)
#
# mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=BinaryAoN(strike=1.), discount=np.exp(-0.06))
# mc_stats.print()
#
# steps = 600
# ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
# gbm_approx = GbmLinear(basis=[basis_1, basis_2, basis_3], time_points=ts, mu=gbm.mu, sigma=gbm.sigma)
# new_cv_stats = mc_control_variate(num_trials=(500, 5000), simple_solver=solver, approximator=gbm_approx,
#                                   payoff=BinaryAoN(strike=1.), discount=np.exp(-0.06))
# new_cv_stats.print()
