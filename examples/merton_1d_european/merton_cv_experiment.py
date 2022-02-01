import torch
from sde_mc import *
import argparse
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('imports successful')
print(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', default=dir_path+'/results', type=str)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--init_trials', default=10000, type=int)
    parser.add_argument('--bs', default=2000, type=int)
    parser.add_argument('--sim_bs', default=100000, type=int)
    parser.add_argument('--train', default=10000, type=int)
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
        f = Mlp(2, [50, 50, 50], 1, batch_norm=False, batch_norm_init=False, device=device)
        g = Mlp(2, [50, 50, 50], 1, batch_norm=False, batch_norm_init=False, device=device)
        adam = torch.optim.Adam(list(f.parameters()) + list(g.parameters()))
        euro_call = EuroCall(strike)
        solver.num_steps = 100
        
        train_time_start = time.time()
        train_dataloader = simulate_adapted_data(args.train, solver, euro_call, short_rate, bs=1000)
        _ = train_adapted_control_variates([f, g], adam, train_dataloader, solver, short_rate, 10, True)
        train_time_end = time.time()
        
        solver.num_steps = 1000
        mc_stats = mc_apply_cvs([f, g], solver, args.init_trials, euro_call, short_rate, sim_bs=args.init_trials, bs=args.bs)
        ratio = (mc_stats.sample_std * 2 / args.tol) ** 2
        trials = np.ceil(ratio * args.init_trials)

        trials = args.bs * np.ceil(trials / args.bs)

        mc_stats = mc_apply_cvs([f, g], solver, trials, euro_call, short_rate, sim_bs=args.sim_bs, bs=args.bs)

        solutions.append(mc_stats.sample_mean)
        errors.append(mc_stats.sample_std * 2)
        num_trials.append(trials)
        times.append(mc_stats.time_elapsed + (train_time_end - train_time_start))

    print('experiment successful')

    data = pd.DataFrame(
        {'strike': strikes, 'mean': solutions, 'conf_interval': errors, 'time': times, 'trials': num_trials})
    data.to_csv(args.dir + '/' + args.fname)
    print('save successful')
