import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .varred import train_control_variates, apply_control_variates, train_diffusion_control_variate, \
    apply_diffusion_control_variate
from .nets import NormalJumpsPathData, NormalPathData
from .helpers import partition, mc_estimates


class MCStatistics:
    """A class to store relevant Monte Carlo statistics"""

    def __init__(self, sample_mean, sample_std, time_elapsed, paths=None, payoffs=None, normals=None):
        """
        :param sample_mean: torch.tensor
            The mean of the samples

        :param sample_std: torch.tensor
            The standard deviation of the samples / sqrt(num_trials)

        :param time_elapsed: float
            The total time taken for the MC simulation

        :param paths: torch.tensor, default = None
            The sample paths generated from the SDE

        :param payoffs: torch.tensor, default = None
            All payoffs generated from the SDE
        """
        self.sample_mean = sample_mean.item()
        self.sample_std = sample_std.item()
        self.time_elapsed = time_elapsed
        self.paths = paths
        self.payoffs = payoffs
        self.normals = normals

    def print(self, num_std=2):
        """Prints the mean, confidence interval, and time taken

        :param num_std: float
            The coefficient corresponding to fiducial probabilities - e.g. num_std = 2
            corresponds to a 95% confidence interval
        """
        print('Mean: {:.5f}  +/- {:.5f}     Time taken (s): {:.2f}'.format(self.sample_mean, self.sample_std * num_std,
                                                                           self.time_elapsed))


def mc_simple(num_trials, sde_solver, payoff, discount=1, bs=None, return_normals=False):
    """Run Monte Carlo simulations of an SDE

    :param num_trials: int
        The number of MC simulations

    :param sde_solver: SdeSolver
        The solver for the SDE

    :param payoff: Option
        The payoff function applied to the terminal value - see the Option class

    :param discount: float, default = 1
        A discount factor to be applied to the payoffs

    :param bs: int, default = None
        The batch size. When None all trials will be done simultaneously. When a bs is specified the payoffs
        and paths will not be recorded

    :param return_normals: bool, default = False
        If True, passes return_normals to the Euler method of the SDE solver. Returns the normal random variables
        used in the numerical integration

    :return: MCStatistics
        The relevant statistics from the MC simulation - see the MCStatistics class
    """
    if not bs:
        start = time.time()
        out, normals = sde_solver.solve(bs=num_trials, return_normals=return_normals)
        spots = out[:, sde_solver.num_steps]
        payoffs = payoff(spots) * discount

        mn = payoffs.mean()
        sd = payoffs.std() / np.sqrt(num_trials)
        end = time.time()
        tt = end - start

        return MCStatistics(mn, sd, tt, out, payoffs, normals)
    else:
        remaining_trials = num_trials
        sample_sum, sample_sum_sq = 0.0, 0.0
        start = time.time()
        while remaining_trials:
            if remaining_trials < bs:
                bs = remaining_trials

            remaining_trials -= bs
            out, normals = sde_solver.solve(bs=bs, return_normals=False)
            spots = out[:, sde_solver.num_steps]
            payoffs = payoff(spots) * discount

            sample_sum += payoffs.sum()
            sample_sum_sq += (payoffs**2).sum()

        mn = sample_sum / num_trials
        sd = torch.sqrt((sample_sum_sq/num_trials - mn**2) * (num_trials / (num_trials-1))) / np.sqrt(num_trials)
        end = time.time()
        tt = end-start
        return MCStatistics(mn, sd, tt)


def mc_control_variates(models, opt, solver, trials, steps, payoff, discounter, sim_bs=(100000, 100000),
                        bs=(1000, 1000), epochs=10):
    # Config
    jump_mean = solver.sde.jump_mean()
    rate = solver.sde.jump_rate()
    train_trials, test_trials = trials
    train_steps, test_steps = steps
    train_bs, test_bs = bs
    train_sim_bs, test_sim_bs = sim_bs
    solver.num_steps = train_steps

    train_time_points = partition(solver.time, train_steps, ends='left', device=solver.device)
    test_time_points = partition(solver.time, test_steps, ends='left', device=solver.device)
    train_ys, test_ys = discounter(train_time_points), discounter(test_time_points)

    # Training
    train_start = time.time()
    train_dataloader = simulate_data(train_trials, solver, payoff, discounter, bs=train_bs)
    if solver.has_jumps:
        _, losses = train_control_variates(models, opt, train_dataloader, jump_mean, rate, train_time_points,
                                           train_ys, epochs)
    else:
        _, losses = train_diffusion_control_variate(models, opt, train_dataloader, train_time_points, train_ys, epochs)
    train_end = time.time()
    train_time = train_end - train_start

    # Inference
    start_test = time.time()
    solver.num_steps = test_steps
    run_sum, run_sum_sq = 0, 0
    trials_remaining = test_trials
    while trials_remaining > 0:
        batch_size = min(test_sim_bs, trials_remaining)
        trials_remaining -= batch_size
        test_dataloader = simulate_data(batch_size, solver, payoff, discounter, bs=test_bs, inference=True)
        if solver.has_jumps:
            x, y = apply_control_variates(models, test_dataloader, jump_mean, rate, test_time_points, test_ys)
        else:
            x, y = apply_diffusion_control_variate(models, test_dataloader, test_time_points, test_ys)
        run_sum += x
        run_sum_sq += y

    mn, var = mc_estimates(run_sum, run_sum_sq, test_trials)
    sd = var.sqrt() / torch.tensor(test_trials).sqrt()
    end_test = time.time()
    test_time = end_test - start_test
    return MCStatistics(mn, sd, train_time+test_time)


def simulate_data(trials, solver, payoff, discounter, bs=1000, inference=False):
    if inference:
        assert not trials % bs, 'Batch size should partition total trials evenly'
    mc_stats = mc_simple(trials, solver, payoff, discounter(solver.time), return_normals=True)
    if solver.has_jumps:
        paths, (paths_no_jumps, normals, jumps) = mc_stats.paths, mc_stats.normals
        payoffs = mc_stats.payoffs
        dset = NormalJumpsPathData(paths, paths_no_jumps, payoffs, normals.squeeze(-1), jumps)
    else:
        paths, (_, normals, _) = mc_stats.paths, mc_stats.normals
        payoffs = mc_stats.payoffs
        dset = NormalPathData(paths, payoffs, normals.squeeze(-1))
    return DataLoader(dset, batch_size=bs, shuffle=not inference, drop_last=not inference)
