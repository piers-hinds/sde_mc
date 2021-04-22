import time
import numpy as np


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


def mc_simple(num_trials, sde_solver, payoff, discount=1):
    """A simple Monte Carlo method applied to an SDE
    :param num_trials: int, the number of MC simulations
    :param sde_solver: SdeSolver, the solver for the SDE
    :param payoff: function, a payoff function to be applied to the process at the final time step, it must be able to
    be applied across a torch.tensor
    :param discount: float (optional), a discount factor to be applied to the payoffs
    :return: MCStatistics, the relevant statistics from the MC simulation - see MCStatistics class
    """
    start = time.time()
    out = sde_solver.euler(bs=num_trials)
    spots = out[:, sde_solver.num_steps, :].squeeze(-1)
    payoffs = payoff(spots, 1) * discount
    end = time.time()

    mn = payoffs.mean()
    sd = payoffs.std() / np.sqrt(num_trials)
    tt = end - start

    return MCStatistics(mn, sd, tt, out, payoffs)
