from sde_mc import *
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# Model params:
r = 0.02
vol = 0.3
alpha = -0.05
gamma = 0.3
rate = 2.
spot = 1.

x0 = torch.tensor([spot], device=device)
jump_mean = (np.exp(alpha + 0.5*gamma**2)) - 1

# Scheme params
expiry = 3
steps = 100
trials = 10000
time_points = partition(expiry, steps, device=device)

# Option and discounter
euro_call = EuroCall(strike=1.)
discounter = ConstantShortRate(r=r)


gbm = Gbm(mu=r, sigma=0.4, init_value=x0, dim=1)
solver = SdeSolver(gbm, expiry, steps)
bcv = Mlp(gbm.dim + 1, [50, 50, 50], 1, activation=nn.ReLU).to(device)
adam = optim.Adam(bcv.parameters())
mc_stats = mc_control_variates(bcv, adam, solver, (1000, 4000), steps=(30, 300), payoff=euro_call,
                               discounter=discounter, bs=(1000, 1000))
mc_stats.print()

mc_stats = mc_simple(4000, solver, euro_call, discounter(solver.time))
mc_stats.print()

print(round(bs_call(1, 1, 3, 0.02, 0.4), 6))



# # Merton SDE and SdeSolver
# pure_jump = Merton(r, vol, rate, alpha, gamma, x0, dim=1)
# dim = pure_jump.dim
# solver = SdeSolver(pure_jump, expiry, steps, device=device)
#
# # Analytical price
# true_price = merton_jump_call(spot, 1, expiry, r, vol, alpha, gamma, rate)
# print(round(true_price, 6))
#
# # Define modules for each control variate
# brownian_control_variate = Mlp(dim + 1, [50, 50, 50], 1, activation=nn.ReLU).to(device)
# jumps_control_variate = Mlp(dim + 1, [50, 50, 50], 1, activation=nn.ReLU).to(device)
#
# # Joint optimizer
# jump_mean = solver.sde.jump_mean()
# adam = optim.Adam(list(brownian_control_variate.parameters()) + list(jumps_control_variate.parameters()))
# mc_stats = mc_control_variates([brownian_control_variate, jumps_control_variate], adam, solver, trials=(1000, 4000),
#                                steps=(30, 300), payoff=euro_call, discounter=discounter, bs=(1000, 1000), epochs=10)
# mc_stats.print()
#
#
# trials = 100000
# mc_stats = mc_simple(trials, solver, euro_call, discount=discounter(solver.time))
# mc_stats.print()