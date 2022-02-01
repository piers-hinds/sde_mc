import torch
from sde_mc import *
import argparse
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('imports successful')
print('running on: ', device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', default=dir_path+'/results', type=str)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--init_trials', default=10000, type=int)
    parser.add_argument('--bs', default=10000, type=int)
    parser.add_argument('--fname', default='results.csv', type=str)

    args = parser.parse_args()
    merton = Merton(0.02, 0.2, 1, -0.05, 0.3, torch.tensor([1.]), dim=1)
    solver = JumpEulerSolver(merton, 3, 1000, device=device)
    short_rate = ConstantShortRate(merton.mu)

    solutions = []
    errors = []
    times = []
    num_trials = []
    strikes = torch.linspace(0.70, 1.3, 7)

    for strike in strikes:
        euro_call = EuroCall(strike)
        mc_stats = mc_simple(args.init_trials, solver, euro_call, short_rate)
        ratio = (mc_stats.sample_std * 2 / args.tol) ** 2
        trials = np.ceil(ratio * args.init_trials)

        mc_stats = mc_simple(trials, solver, euro_call, short_rate, bs=args.bs)

        solutions.append(mc_stats.sample_mean)
        errors.append(mc_stats.sample_std * 2)
        num_trials.append(trials)
        times.append(mc_stats.time_elapsed)

    print('experiment successful')

    data = pd.DataFrame(
        {'strike': strikes, 'mean': solutions, 'conf_interval': errors, 'time': times, 'trials': num_trials})
    data.to_csv(args.dir + '/' + args.fname)
    print(args.dir + '/' + args.fname)
    print('save successful')
