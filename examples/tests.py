from sde_mc import *


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
Ys = discounter(time_points)

# Merton SDE and SdeSolver
pure_jump = Merton(r - rate*jump_mean, vol, x0, rate, alpha, gamma)
dim = pure_jump.dim
solver = JumpSolver(pure_jump, expiry, steps, device=device)

# Analytical price
true_price = merton_jump_call(spot, 1, expiry, r, vol, alpha, gamma, rate)
print(round(true_price, 6))

# Define modules for each control variate
brownian_control_variate = Mlp(dim + 1, [50, 50, 50], 1, activation=nn.ReLU).to(device)
jumps_control_variate = Mlp(dim + 1, [50, 50, 50], 1, activation=nn.ReLU).to(device)

# Joint optimizer
adam = optim.Adam(list(brownian_control_variate.parameters()) + list(jumps_control_variate.parameters()))
mc_stats = mc_control_variates([brownian_control_variate, jumps_control_variate], adam, solver, trials=(1000, 4000),
                               steps=(30, 300), payoff=euro_call, discounter=discounter, jump_mean=jump_mean,
                               rate=rate, bs=(1000, 1000), epochs=10)
mc_stats.print()

trials = 4000
bs = 1000
steps = 300
solver.steps = steps
mc_stats = mc_simple(trials, solver, euro_call, discount=discounter(solver.time), bs=bs)
mc_stats.print()