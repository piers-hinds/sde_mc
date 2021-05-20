from sde_mc import *

# Pricing a binary asset-or-nothing option with GBM dynamics for the underlier.
# The control variate is estimated using a feed-forward neural network.


# MC with no control variate
steps = 3000
trials = 20000
gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
paths = solver.euler(2)

mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=BinaryAoN(strike=1.), discount=np.exp(-0.06))
mc_stats.print()

# MC with control variate
steps = 100
ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
net_approx = NetApproximatr(layer_sizes=[3], time_points=ts)
new_cv_stats = mc_control_variate(num_trials=(150, 5000), simple_solver=solver, approximator=net_approx,
                                  payoff=BinaryAoN(strike=1.), discounter=ConstantShortRate(r=0.02), step_factor=30)
new_cv_stats.print()
