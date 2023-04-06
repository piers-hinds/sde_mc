from sde_mc import *
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


gbm = Gbm(0.02, 0.2, torch.tensor([1.]), dim=1)

solver = EulerSolver(gbm, 1, 100)
print(solver.sample_corr_normals((5, 2, 1, 1), torch.tensor(0.1), corr=False))

