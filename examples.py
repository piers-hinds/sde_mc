from sde_mc import *
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial
from abc import ABC, abstractmethod


def constant_rate(t, rate):
    return torch.exp(-rate * t)


my_discount = partial(constant_rate, rate=0.02)


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
# trials = 20000
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
# net_approx = NetApproximator(layer_sizes=[3], time_points=ts)
# new_cv_stats = mc_control_variate(num_trials=(150, 5000), simple_solver=solver, approximator=net_approx,
#                                   payoff=BinaryAoN(strike=1.), discounter=my_discount, step_factor=30)
# new_cv_stats.print()


# new examples
# steps = 600
# ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
# net_approx = GbmNet(layer_sizes=[20, 10], time_points=ts)
#
# mc_stats = mc_simple(500, sde_solver=solver, payoff=BinaryAoN(strike=1.), discount=np.exp(-0.06))
# paths = mc_stats.paths
# payoffs = mc_stats.payoffs
# net_approx.fit(paths, payoffs)
#
# xs = torch.linspace(0.5, 2, 100)
# t = torch.tensor(2.99).repeat(len(xs))
# input = torch.stack([t, xs], dim=1)
# out = net_approx.mlp(input)
#
# plt.scatter(paths[:, 598], payoffs)
# plt.plot(xs, out.detach())
# plt.show()






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
#
# mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=BinaryAoN(strike=1.), discount=np.exp(-0.06))
# mc_stats.print()

# steps = 600
# ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
# gbm_approx = LinearApproximator(basis=[basis_1, basis_2, basis_3], time_points=ts)
# new_cv_stats = mc_control_variate(num_trials=(500, 5000), simple_solver=solver, approximator=gbm_approx,
#                                   payoff=BinaryAoN(strike=1.), discounter=my_discount)
# new_cv_stats.print()


# steps = 100
# ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
# x0 = torch.tensor([np.log(1), np.log(0.2**2)])
# heston = Heston(r=0.02, kappa=0.2, theta=0.2**2, xi=0.1, rho=-0.2, init_value=x0)
# solver = SdeSolver(sde=heston, time=3, num_steps=steps)
# heston_cv = SdeControlVariate(base_sde=heston, control_variate=None, time_points=ts, discounter=my_discount)
#
# mc_stats = mc_simple(num_trials=100, sde_solver=solver, payoff=EuroCall(strike=1., log=True), discount=np.exp(-0.06))
# paths, payoffs = mc_stats.paths, mc_stats.payoffs
#
# heston_approx = GbmNet(time_points=ts, layer_sizes=[10, 10], dim=2)
# heston_approx.fit(paths, payoffs)
# xs = torch.linspace(0.5, 2, 100)
# vs = torch.log(torch.tensor(0.2**2)).repeat(len(xs))
# t = torch.tensor(2.9).repeat(len(xs))
#
# input = torch.stack([t, torch.log(xs), vs], dim=1)
# out = heston_approx.mlp(input)
# d_out = heston_approx(0, torch.tensor(2.9), torch.stack([xs, vs], dim=1))
# print(d_out)
# plt.plot(xs, out.detach())
# plt.show()



# Heston Example with CV:
# steps = 3000
# ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
# x0 = torch.tensor([np.log(1), np.log(0.2**2)])
# heston = Heston(r=0.02, kappa=0.2, theta=0.2**2, xi=0.1, rho=-0.2, init_value=x0)
# solver = SdeSolver(sde=heston, time=3, num_steps=steps)
# mc_stats = mc_simple(num_trials=1000, sde_solver=solver, payoff=EuroCall(strike=1., log=True), discount=np.exp(-0.06))
# mc_stats.print()
#
#
# steps = 600
# ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
# x0 = torch.tensor([np.log(1), np.log(0.2**2)])
# heston = Heston(r=0.02, kappa=0.2, theta=0.2**2, xi=0.1, rho=-0.2, init_value=x0)
# solver = SdeSolver(sde=heston, time=3, num_steps=steps)
# heston_approx = GbmNet(time_points=ts, layer_sizes=[10, 10], dim=2)
# mc_stats = mc_simple(num_trials=3, sde_solver=solver, payoff=EuroCall(strike=1., log=True), discount=np.exp(-0.06))
# paths, payoffs = mc_stats.paths, mc_stats.payoffs
# heston_approx.fit(paths, payoffs)
# heston_cv = SdeControlVariate(base_sde=heston, control_variate=heston_approx, time_points=ts, discounter=my_discount)

# derivs = heston_approx(0, torch.tensor(1.), paths[:, 0])
# print(derivs.unsqueeze(-1))
# sigma_t = torch.transpose(heston.diffusion(0, paths[:, 0]), 1, 2)
# print(torch.matmul(-sigma_t, derivs.unsqueeze(-1)).transpose(1, 2))
#
# heston_cv.reset_control(paths[:, 0])
# print(heston_cv.F)
#print(heston_cv.diffusion(torch.tensor(1.), paths[:, 0]))

# mc_stats_cv = mc_control_variate((100, 1000), simple_solver=solver, approximator=heston_approx,
#                                  payoff=EuroCall(strike=1., log=True), discounter=my_discount)
# mc_stats_cv.print()













