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
solver = SdeSolver(merton, 3, 10)
paths,  (left_paths, normals, jumps) = solver.solve(bs=1, return_normals=True)
print(paths[0])
all_jumps = torch.cat([torch.zeros_like(paths[:, :1, :]), jumps], dim=1)
print(left_paths + (all_jumps) * left_paths)





















