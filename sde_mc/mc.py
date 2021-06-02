import time
import numpy as np
import torch
from .vreduction import SdeControlVariate
from .sde import SdeSolver


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
        out, normals = sde_solver.euler(bs=num_trials, return_normals=return_normals)
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
            out, normals = sde_solver.euler(bs=bs, return_normals=False)
            spots = out[:, sde_solver.num_steps]
            payoffs = payoff(spots) * discount

            sample_sum += payoffs.sum()
            sample_sum_sq += (payoffs**2).sum()

        mn = sample_sum / num_trials
        sd = torch.sqrt((sample_sum_sq/num_trials - mn**2) * (num_trials / (num_trials-1))) / np.sqrt(num_trials)
        end = time.time()
        tt = end-start
        return MCStatistics(mn, sd, tt)


def mc_control_variate(num_trials, simple_solver, approximator, payoff, discounter, step_factor=5, time_points=None,
    bs=None):
    """Run Monte Carlo simulation of an SDE with a control variate

    :param num_trials: tuple (int, int)
        The number of trials for the approximation and the number of trials for the
        control variate monte carlo method

    :param simple_solver: SdeSolver
        The solver for the Sde with no control variate

    :param approximator: SdeApproximator
        The approximator for the solution of the Sde

    :param payoff: Option
        The payoff function applied to the terminal value - see the Option class

    :param discounter: function(float: time)
        The discount process to be applied to the payoff

    :param step_factor: int
        The factor to increase the number of steps used in the initial solver

    :param bs: int
        The batch size for the monte carlo method

    :return: MCStatistics
        The relevant statistics from the MC simulation - see the MCStatistics class
    """
    if time_points is None:
        time_points = approximator.time_points
    simple_trials, cv_trials = num_trials
    discount = discounter(torch.tensor(simple_solver.time))
    start = time.time()
    simple_stats = mc_simple(simple_trials, simple_solver, payoff, discount)
    approximator.fit(simple_stats.paths, simple_stats.payoffs)
    cv_sde = SdeControlVariate(base_sde=simple_solver.sde, control_variate=approximator,
                               time_points=time_points, discounter=discounter)
    cv_solver = SdeSolver(sde=cv_sde, time=3, num_steps=simple_solver.num_steps*step_factor,
                          device=simple_solver.device)

    def cv_payoff(spot):
        return discount * payoff(spot[:, :simple_solver.sde.dim]) + spot[:, simple_solver.sde.dim]

    cv_stats = mc_simple(cv_trials, cv_solver, cv_payoff, discount=1, bs=bs)
    print('Time for final MC:', cv_stats.time_elapsed)
    end = time.time()
    cv_stats.time_elapsed = end-start
    return cv_stats
