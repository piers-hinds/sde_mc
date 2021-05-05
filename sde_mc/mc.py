import time
import numpy as np
import torch
from .vreduction import SdeControlVariate
from .sde import SdeSolver


class MCStatistics:
    """A class to store relevant Monte Carlo statistics"""

    def __init__(self, sample_mean, sample_std, time_elapsed, paths=None, payoffs=None):
        """
        :param sample_mean: torch.tensor, the mean of the samples
        :param sample_std: torch.tensor, the standard deviation of the samples / sqrt(num_trials)
        :param time_elapsed: float, the time taken for the MC simulation
        :param paths: torch.tensor (optional), the sample paths generated from the SDE
        :param payoffs: torch.tensor (optional), all payoffs generated from the SDE
        """
        self.sample_mean = sample_mean.item()
        self.sample_std = sample_std.item()
        self.time_elapsed = time_elapsed
        self.paths = paths
        self.payoffs = payoffs

    def print(self, num_std=2):
        """Prints the mean, confidence interval, and time taken
        :param num_std: float, the coefficient corresponding to fiducial probabilities - e.g. num_std = 2
        corresponds to a 95% confidence interval
        :return: None, prints output
        """
        print('Mean: {:.5f}  +/- {:.5f}     Time taken (s): {:.2f}'.format(self.sample_mean, self.sample_std * num_std,
                                                                           self.time_elapsed))


def mc_simple(num_trials, sde_solver, payoff, discount=1, bs=None):
    """A simple Monte Carlo method applied to an SDE
    :param num_trials: int, the number of MC simulations
    :param sde_solver: SdeSolver, the solver for the SDE
    :param payoff: function, a payoff function to be applied to the process at the final time step, it must be able to
    be applied across a torch.tensor
    :param discount: float (optional), a discount factor to be applied to the payoffs
    :param bs: int, the batch size. When None all trials will be done simultaneously. When a bs is specified the payoffs
    and paths will not be recorded
    :return: MCStatistics, the relevant statistics from the MC simulation - see MCStatistics class
    """
    if not bs:
        start = time.time()
        out = sde_solver.euler(bs=num_trials)
        spots = out[:, sde_solver.num_steps]
        payoffs = payoff(spots) * discount

        mn = payoffs.mean()
        sd = payoffs.std() / np.sqrt(num_trials)
        end = time.time()
        tt = end - start

        return MCStatistics(mn, sd, tt, out, payoffs)
    else:
        remaining_trials = num_trials
        sample_sum, sample_sum_sq = 0.0, 0.0
        start = time.time()
        while remaining_trials:
            if remaining_trials < bs:
                bs = remaining_trials

            remaining_trials -= bs
            out = sde_solver.euler(bs=bs)
            spots = out[:, sde_solver.num_steps, :].squeeze(-1)
            payoffs = payoff(spots, 1) * discount

            sample_sum += payoffs.sum()
            sample_sum_sq += (payoffs**2).sum()

        mn = sample_sum / num_trials
        sd = torch.sqrt((sample_sum_sq/num_trials - mn**2) * (num_trials / (num_trials-1))) / np.sqrt(num_trials)
        end = time.time()
        tt = end-start
        return MCStatistics(mn, sd, tt)


def mc_control_variate(num_trials, simple_solver, approximator, payoff, discount, bs=None):
    simple_trials, cv_trials = num_trials
    start = time.time()
    simple_stats = mc_simple(simple_trials, simple_solver, payoff, discount)
    approximator.fit(simple_stats.paths, simple_stats.payoffs)
    cv_sde = SdeControlVariate(base_sde=simple_solver.sde, control_variate=approximator,
                               time_points=approximator.time_points)
    cv_solver = SdeSolver(sde=cv_sde, time=3, num_steps=simple_solver.num_steps*5, device=simple_solver.device)

    def cv_payoff(spot):
        return discount * payoff(spot[:, :simple_solver.sde.dim]) + spot[:, simple_solver.sde.dim]

    cv_stats = mc_simple(cv_trials, cv_solver, cv_payoff, discount=1, bs=bs)
    print(cv_stats.time_elapsed)
    end = time.time()
    cv_stats.time_elapsed = end-start
    return cv_stats
