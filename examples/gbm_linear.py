from sde_mc import *

# Pricing a binary asset-or-nothing option with GBM dynamics for the underlier.
# This example shows MC with a control variate is 3-4x faster than MC without a control variate. The control variate
# is estimated using linear regression on a chosen set of basis functions.

# MC without control variates for comparison
steps = 3000
trials = 70000

gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
solver = SdeSolver(sde=gbm, time=3, num_steps=steps)

mc_stats = mc_simple(num_trials=trials, sde_solver=solver, payoff=BinaryAoN(strike=1.), discount=np.exp(-0.06))
mc_stats.print()

# MC with control variates
steps = 600
ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]), dim=1)
solver = SdeSolver(sde=gbm, time=3, num_steps=steps)
gbm_approx = LinearApproximator(basis=[basis_1, basis_2, basis_3], time_points=ts)
new_cv_stats = mc_control_variate(num_trials=(500, 5000), simple_solver=solver, approximator=gbm_approx,
                                  payoff=BinaryAoN(strike=1.), discounter=ConstantShortRate(r=0.02))
new_cv_stats.print()
