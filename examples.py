from sde_mc import *

S = 1.
K = 1
T = 3
r = 0.02
sigma = 0.3
alpha = -0.05
gamma = 0.3
rate = 2

call = EuroCall(strike=K)

merton = Merton(0.02, 0.3, 2, -0.05, 0.3, torch.tensor([1.]), dim=1)
solver = JumpEulerSolver(merton, 3, 10)
mc_stats = mc_simple(10, solver, call)
print(mc_stats)

gbm = DoubleGbm(r, 0.1, 0.5, torch.tensor([1.]), dim=1)
solver = EulerSolver(gbm, 3, 1000)
mc_stats = mc_simple(100000, solver, call, ConstantShortRate(0.02), bs=100000)
print(mc_stats)

gbm = Gbm(r, np.sqrt(0.1**2 + 0.5**2), torch.tensor([1.]), dim=1)
solver = EulerSolver(gbm, 3, 1000)
mc_stats = mc_simple(100000, solver, call, ConstantShortRate(0.02), bs=100000)
print(mc_stats)























