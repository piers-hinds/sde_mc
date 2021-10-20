import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .varred import train_diffusion_control_variate, apply_diffusion_control_variate, train_adapted_control_variates, \
    apply_adapted_control_variates
from .nets import NormalJumpsPathData, NormalPathData, AdaptedPathData
from .helpers import partition, mc_estimates
from .options import ConstantShortRate


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

    def __str__(self):
        """Prints the mean, 95% confidence interval, and time taken
        """
        return 'Mean: {:.6f}  +/- {:.6f}     Time taken (s): {:.2f}'.format(self.sample_mean, self.sample_std * 2,
                                                                            self.time_elapsed)


def mc_simple(num_trials, sde_solver, payoff, discounter=None, bs=None, return_normals=False, payoff_time='terminal'):
    """Run Monte Carlo simulations of a functional of an SDE's terminal value

    :param num_trials: int
        The number of MC simulations

    :param sde_solver: SdeSolver
        The solver for the SDE

    :param payoff: Option
        The payoff function applied to the terminal value - see the Option class

    :param discounter: callable function of time (default = None)
        A function which returns the discount factor to be applied to the payoffs

    :param bs: int (default = None)
        The batch size. When None all trials will be done simultaneously. When a batch size is specified the payoffs
        and paths will not be recorded

    :param return_normals: bool (default = False)
        If True returns the normal random variables used in the numerical integration

    :param payoff_time: string (default = 'terminal')
        The index (time-step) at which to evaluate the payoff, defaults to the terminal time

    :return: MCStatistics
        The relevant statistics from the MC simulation - see the MCStatistics class
    """
    if discounter is None:
        discounter = ConstantShortRate(r=0.0)

    if payoff_time == 'terminal':
        payoff_index = sde_solver.num_steps

    if not bs:
        start = time.time()
        out, normals = sde_solver.solve(bs=num_trials, return_normals=return_normals)
        if payoff_time == 'adapted':
            payoff_index = normals[3]
        spots = out[:, payoff_index]
        payoffs = payoff(spots) * discounter(sde_solver.time_interval)

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
            if payoff_time == 'adapted':
                payoff_index = normals[3]
            spots = out[:, payoff_index]
            payoffs = payoff(spots) * discounter(sde_solver.time_interval)

            sample_sum += payoffs.sum()
            sample_sum_sq += (payoffs**2).sum()

        mn = sample_sum / num_trials
        sd = torch.sqrt((sample_sum_sq/num_trials - mn**2) * (num_trials / (num_trials-1))) / np.sqrt(num_trials)
        end = time.time()
        tt = end-start
        return MCStatistics(mn, sd, tt)


def mc_control_variates(models, opt, solver, trials, steps, payoff, discounter, sim_bs=(1e5, 1e5),
                        bs=(1000, 1000), epochs=10, print_losses=True, no_train=False):
    """Monte Carlo simulation of a functional of an SDE's terminal value with neural control variates

    Generates initial trajectories and payoffs on which regression is performed to find optimal control variates (a
    coarser time grid can be used). Then the control variates can be used on newly generated trajectories to
    significantly reduce variance.

    :param models: list of nn.Module
        The neural networks used to approximate the control variates for the brownian motion and the Poisson process

    :param opt: optimizer
        Optimizer from torch.optim used to step the parameters in models

    :param solver: SdeSolver
        The solver for the SDE

    :param trials: (int, int)
        Trials for the initial coarser simulation (training) and for the finer simulation (inference)

    :param steps: (int, int)
         Number of steps for the initial coarser simulation (training) and for the finer simulation (inference)

    :param payoff: Option
        The payoff function to be applied to the terminal value of the simulated SDE

    :param discounter: Callable function of time
        The discount function which returns the discount to be applied to the payoffs

    :param sim_bs: (int, int) (default = (1e5, 1e5))
        The batch sizes to simulate the trajectories (fine and coarse, respectively)

    :param bs: (int, int) (default = (1e3, 1e3))
        The batch sizes for training and inference, respectively

    :param epochs: int (default = 10)
        Number of epochs in training

    :param print_losses: bool (default = True)
        If True, prints the loss function values during training
        
    :param no_train: bool (default = False)
        If False, doesn't train the network first.

    :return: MCStatistics
        The relevant MC statistics
    """
    assert not solver.has_jumps
    # Config
    train_trials, test_trials = trials
    train_steps, test_steps = steps
    train_bs, test_bs = bs
    train_sim_bs, test_sim_bs = sim_bs
    solver.num_steps = train_steps

    # Training
    train_start = time.time()
    sim_train_control_variates(models, opt, solver, train_trials, payoff, discounter, train_sim_bs, train_bs, epochs)
    train_end = time.time()

    # Inference
    solver.num_steps = test_steps
    mc_stats = mc_apply_cvs(models, solver, test_trials, payoff, discounter, test_sim_bs, test_bs)
    mc_stats.time_elapsed += train_end - train_start

    return mc_stats


def mc_apply_cvs(models, solver, trials, payoff, discounter, sim_bs=1e5, bs=1000):
    """Monte Carlo simulation of a function of SDE's terminal value with applied control variates

    :param models: callable(s)
        The control variate function(s). If multiple, pass in a tuple or list

    :param solver: SdeSolver
        The solver object for the SDE

    :param trials: int
        The number of Monte Carlo trials

    :param payoff: Option
        The payoff function to be applied to the terminal values

    :param discounter: Discounter
        The discounter to be applied to the terminal spot values

    :param sim_bs: int
        The batch size for the simulation of the trajectories

    :param bs: int
        The batch size for applying the control variate(s)

    :return: MCStatisitics
        The relevant stats from the MC simulation
    """
    start_test = time.time()

    run_sum, run_sum_sq = 0, 0
    trials_remaining = trials
    while trials_remaining > 0:
        batch_size = min(sim_bs, trials_remaining)
        trials_remaining -= batch_size
        test_dl = simulate_data(batch_size, solver, payoff, discounter, bs=bs, inference=True)
        x, y = apply_diffusion_control_variate(models, test_dl, solver, discounter)
        run_sum += x
        run_sum_sq += y

    mn, var = mc_estimates(run_sum, run_sum_sq, trials)
    sd = var.sqrt() / torch.tensor(trials).sqrt()
    end_test = time.time()
    test_time = end_test - start_test
    return MCStatistics(mn, sd, test_time)


def mc_adaptive_cv(models, opt, solver, trials, steps, payoff, discounter, sim_bs=(1e4, 1e4), bs=(1000, 1000),
                   epochs=10, print_losses=True, pre_trained=False):
    """Monte Carlo simulation of a functional of an SDE's terminal value with neural control variates

        Generates initial trajectories and payoffs on which regression is performed to find optimal control variates (a
        coarser time grid can be used). Then the control variates can be used on newly generated trajectories to
        significantly reduce variance.

        :param models: list of nn.Module
            The neural networks used to approximate the control variates for the brownian motion and the Poisson process

        :param opt: optimizer
            Optimizer from torch.optim used to step the parameters in models

        :param solver: SdeSolver
            The solver for the SDE

        :param trials: (int, int)
            Trials for the initial coarser simulation (training) and for the finer simulation (inference)

        :param steps: (int, int)
             Number of steps for the initial coarser simulation (training) and for the finer simulation (inference)

        :param payoff: Option
            The payoff function to be applied to the terminal value of the simulated SDE

        :param discounter: Callable function of time
            The discount function which returns the discount to be applied to the payoffs

        :param sim_bs: (int, int) (default = (1e4, 1e4))
            The batch sizes to simulate the trajectories (fine and coarse, respectively)

        :param bs: (int, int) (default = (1e3, 1e3))
            The batch sizes for training and inference, respectively

        :param epochs: int (default = 10)
            Number of epochs in training

        :param print_losses: bool (default = True)
            If True, prints the loss function values during training

        :return: MCStatistics
            The relevant MC statistics
        """
    # Config
    jump_mean = solver.sde.jump_mean()
    rate = solver.sde.jump_rate()
    train_trials, test_trials = trials
    train_steps, test_steps = steps
    train_bs, test_bs = bs
    train_sim_bs, test_sim_bs = sim_bs
    solver.num_steps = train_steps

    # Training
    train_start = time.time()
    if not pre_trained:
        train_dataloader = simulate_adapted_data(train_trials, solver, payoff, discounter, bs=train_bs)
        _ = train_adapted_control_variates(models, opt, train_dataloader, solver, discounter, epochs, print_losses)
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
        test_dataloader = simulate_adapted_data(batch_size, solver, payoff, discounter, bs=test_bs, inference=True)
        x, y = apply_adapted_control_variates(models, test_dataloader, solver, discounter)
        run_sum += x
        run_sum_sq += y

    mn, var = mc_estimates(run_sum, run_sum_sq, test_trials)
    sd = var.sqrt() / torch.tensor(test_trials).sqrt()
    end_test = time.time()
    test_time = end_test - start_test
    return MCStatistics(mn, sd, train_time+test_time)


def mc_multilevel(trials, levels, solver, payoff, discounter, bs=None):
    """Runs a multilevel Monte Carlo simulation of a functional of the terminal value of an SDE

    :param trials: list of ints
        The number of trials for each level

    :param levels: list of ints
        The number of steps for each level which should be increasing and levels[i] should divide levels[i+1]

    :param solver: SdeSolver
        The solver for an SDE

    :param payoff: Option
        The payoff function to be applied to the terminal values of the trajectories

    :param discounter: Callable function of time
        The discount function which returns a discount to be applied to the payoffs

    :param bs: list of ints (default = None)
        The batch sizes for each level

    :return: MCStatistics
        The relevant Monte Carlo statistics
    """

    if bs is None:
        bs = trials
    start = time.time()
    # Config
    exps = torch.zeros((len(levels)))
    vars = torch.zeros(len(levels))
    trial_numbers = torch.tensor(trials)

    # First level
    run_sum, run_sum_sq = 0, 0
    trials_remaining = trials[0]
    while trials_remaining > 0:
        next_batch_size = min(trials_remaining, bs[0])
        solver.num_steps = levels[0]
        paths, _ = solver.solve(bs=next_batch_size)
        terminals = payoff(paths[:, solver.num_steps, :]) * discounter(solver.time_interval)
        run_sum += terminals.sum()
        run_sum_sq += (terminals * terminals).sum()
        trials_remaining -= next_batch_size

    exps[0], vars[0] = mc_estimates(run_sum, run_sum_sq, trials[0])

    # Other levels
    pairs = [(levels[i + 1], levels[i]) for i in range(0, len(levels) - 1)]
    for i, pair in enumerate(pairs):
        run_sum, run_sum_sq = 0, 0
        trials_remaining = trials[i + 1]
        while trials_remaining > 0:
            next_batch_size = min(trials_remaining, bs[i + 1])
            (paths_fine, paths_coarse), _ = solver.multilevel_euler(next_batch_size, pair)
            terminals = discounter(solver.time_interval) * (payoff(paths_fine[:, pair[0]]) -
                                                            payoff(paths_coarse[:, pair[1]]))
            run_sum += terminals.sum()
            run_sum_sq += (terminals * terminals).sum()
            trials_remaining -= next_batch_size
        exps[i + 1], vars[i + 1] = mc_estimates(run_sum, run_sum_sq, trials[i + 1])

    total_sd = (vars / trial_numbers).sum().sqrt()
    total_mean = exps.sum()
    end = time.time()
    return MCStatistics(total_mean, total_sd, end - start)


def simulate_data(trials, solver, payoff, discounter, bs=1000, inference=False):
    """Simulates trajectories of an SDE and returns the trajectories, payoffs and random variables in a DataLoader
    which can be used for training or inference"""

    if inference:
        assert not trials % bs, 'Batch size should partition total trials evenly'
    mc_stats = mc_simple(trials, solver, payoff, discounter, return_normals=True)
    paths, normals = mc_stats.paths, mc_stats.normals
    payoffs = mc_stats.payoffs
    dset = NormalPathData(paths, payoffs, normals)
    return DataLoader(dset, batch_size=bs, shuffle=not inference, drop_last=not inference)


def simulate_adapted_data(trials, solver, payoff, discounter, bs=1000, inference=False):
    mc_stats = mc_simple(trials, solver, payoff, discounter, return_normals=True, payoff_time='adapted')
    if solver.has_jumps:
        paths, (normals, time_paths, left_paths, total_steps, jump_paths) = mc_stats.paths, mc_stats.normals
        payoffs = mc_stats.payoffs
        dset = AdaptedPathData(paths[:, :total_steps+1], payoffs, normals[:, :total_steps], left_paths[:, :total_steps+1],
                                       time_paths[:, :total_steps+1], jump_paths[:, :total_steps+1], total_steps)
    return DataLoader(dset, batch_size=bs, shuffle=not inference, drop_last=not inference)


def sim_train_control_variates(models, opt, solver, trials, payoff, discounter, sim_bs, bs, epochs=10,
                               print_losses=True):
    train_dl = simulate_data(trials, solver, payoff, discounter, bs=bs)
    _, losses = train_diffusion_control_variate(models, opt, train_dl, solver, discounter, epochs, print_losses)


def get_optimal_trials(trials, levels, epsilon, solver, payoff, discounter):
    """Finds the optimal number of trials at each level for the MLMC method (for a given tolerance)"""

    vars = torch.zeros(len(levels))
    pairs = [(levels[i + 1], levels[i]) for i in range(0, len(levels) - 1)]
    step_sizes = solver.time_interval / torch.tensor(levels)
    solver.num_steps = levels[0]
    paths, _ = solver.solve(bs=trials)
    discounted_payoffs = payoff(paths[:, solver.num_steps, :]) * discounter(solver.time_interval)
    var = discounted_payoffs.var()
    vars[0] = var

    for i, pair in enumerate(pairs):
        (paths_fine, paths_coarse), _ = solver.multilevel_euler(trials, pair)
        terminals = discounter(solver.time_interval) * (payoff(paths_fine[:, pair[0]]) -
                                                        payoff(paths_coarse[:, pair[1]]))
        vars[i + 1] = terminals.var()

    sum_term = (vars / step_sizes).sqrt().sum()
    optimal_trials = (2 / (epsilon * epsilon)) * (vars * step_sizes).sqrt() * sum_term
    return optimal_trials.ceil().long().tolist()
