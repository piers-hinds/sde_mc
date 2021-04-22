from options import aon_payoff, aon_true
from sde import Gbm, SdeSolver
import torch
import time


steps = 800
trials = 10000

gbm = Gbm(mu=0.02, sigma=0.2, init_value=torch.tensor([1.0]))
test = SdeSolver(sde=gbm, time=3, num_steps=steps, dimension=1)


start = time.time()
out = test.euler(bs=trials)
spots = out[:, steps, :].squeeze(-1)
payoffs = aon_payoff(spots, 1) * torch.exp(torch.tensor(-0.06))
end = time.time()
print(end-start)
print('True value: ', aon_true(1, 1, 0.02, 0.2, 3))
print(torch.mean(payoffs), "+/-", 2 * torch.std(payoffs) / torch.sqrt(torch.tensor(trials)))
