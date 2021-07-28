from sde_mc import *

# Pricing a European call option under the Heston model:

# MC with no control variate
steps = 3000
x0 = torch.tensor([np.log(1), np.log(0.2**2)])
heston = Heston(r=0.02, kappa=0.2, theta=0.2**2, xi=0.1, rho=-0.2, init_value=x0)
solver = SdeSolver(sde=heston, time=3, num_steps=steps)
mc_stats = mc_simple(num_trials=1000, sde_solver=solver, payoff=EuroCall(strike=1., log=True), discount=np.exp(-0.06))
mc_stats.print()


# MC with a control variate
steps = 600
ts = torch.tensor([3*i / steps for i in range(1, steps+1)])
x0 = torch.tensor([np.log(1), np.log(0.2**2)])
heston = Heston(r=0.02, kappa=0.2, theta=0.2**2, xi=0.1, rho=-0.2, init_value=x0)
solver = SdeSolver(sde=heston, time=3, num_steps=steps)
heston_approx = NetApproximator(time_points=ts, layer_sizes=[10, 10], dim=2)
mc_stats_cv = mc_control_variate((100, 1000), simple_solver=solver, approximator=heston_approx,
                                 payoff=EuroCall(strike=1., log=True), discounter=ConstantShortRate(r=0.02))
mc_stats_cv.print()