from sde_mc import *
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial
from abc import ABC, abstractmethod


# What takes time in NN eval? 1) Evaluating the gradients or 2) Just updating F
# If 1, think about alternatives to current implementation of NN, possible the autograd function, no problem in
# manual grad calculation like in linear case
# If 2, make F update faster (Should be possible since linear method has no problem)

print('Original method:')
steps = 100
ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
net_approx = NetApproximator(layer_sizes=[10, 5], time_points=ts, discounter=ConstantShortRate(r=0.02))
new_cv_stats = mc_control_variate(num_trials=(150, 5000), simple_solver=solver, approximator=net_approx,
                                  payoff=BinaryAoN(strike=1.), discounter=ConstantShortRate(r=0.02), step_factor=30)
new_cv_stats.print()

x = torch.tensor([[4., 6.], [1., 2.5]])
print(x.max(1))
rb = Rainbow(strike=4.)
print(rb(x))



# print('New method:')
# steps = 100
# ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
# gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
# solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
# net_approx = NetApproximator(layer_sizes=[10], time_points=ts, discounter=ConstantShortRate(r=0.02))
# net_approx.mlp = MlpTest(input_size=2, layer_sizes=[10, 5], output_size=1)
# new_cv_stats = mc_control_variate(num_trials=(150, 5000), simple_solver=solver, approximator=net_approx,
#                                   payoff=BinaryAoN(strike=1.), discounter=ConstantShortRate(r=0.02), step_factor=30)
# new_cv_stats.print()


# Basket option on multi-dimensional GBM
# steps = 3000
# trials = 1000
# dim = 4
#
# x0 = torch.tensor(1.).repeat(dim)
# print(x0.unsqueeze(0))
# multi_gbm = Gbm(mu=0.02, sigma=torch.tensor([0.1, 0.2, 0.15, 0.3]), init_value=x0, dim=dim)
# solver = SdeSolver(sde=multi_gbm, time=3, num_steps=steps)
# mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=Basket(strike=1.), discount=np.exp(-0.06))
# mc_stats.print()
#
#
# steps = 300
# sf = int(3000 / steps)
# ts = torch.tensor([i * 3 / steps for i in range(1, steps+1)])
# multi_gbm = Gbm(mu=0.02, sigma=torch.tensor([0.1, 0.2, 0.15, 0.3]), init_value=x0, dim=dim)
# solver = SdeSolver(sde=multi_gbm, time=3, num_steps=steps)
# gbm_approx = NetApproximator(time_points=ts, layer_sizes=[15, 15], dim=dim, epochs=10, bs=1024)
# mc_stats_cv = mc_control_variate((300, 1000), simple_solver=solver, approximator=gbm_approx, payoff=Basket(strike=1.),
#                                  discounter=ConstantShortRate(r=0.02), step_factor=sf)
# mc_stats_cv.print()

























