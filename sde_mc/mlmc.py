import torch
import time
from .helpers import mc_estimates
from mc import MCStatistics


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