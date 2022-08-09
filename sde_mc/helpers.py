import torch
import numpy as np
from scipy.integrate import quad


def partition(interval, steps, ends='right', device='cpu'):
    """Alternative to torch.linspace to make it easier to choose whether endpoints are included

    :param interval: float
        The endpoint of the interval

    :param steps: int
        Number of steps

    :param ends: str (in ['left', 'right', 'both', None]) (default = 'right')
        The endpoints to include in the partition. For example, 'right' includes only the last endpoint and not the
        first

    :param device: str (default 'cpu')
        The device for the tensor

    :return: torch.tensor
    """
    assert ends in ['right', 'left', 'both', 'none']
    if ends in ['right', 'both']:
        right = steps + 1
    else:
        right = steps
    if ends in ['left', 'both']:
        left = 0
    else:
        left = 1
    return torch.tensor([interval * i / steps for i in range(left, right)], device=device)


def solve_quadratic(coefficients):
    """Solves quadratics across batches, returns only maximum root

    :param coefficients: 3-tuple of torch.tensors (a, b, c)
        Each of (a, b, c) should be of length batch_size and all on the same device. (a, b, c) corresponds to the usual
        quadratic coefficients

    :return: torch.tensor of length batch_size
    """
    a, b, c = coefficients
    sol1 = (- b + torch.sqrt(b * b - 4 * a * c)) / (2 * a)
    sol2 = (- b - torch.sqrt(b * b - 4 * a * c)) / (2 * a)
    return torch.maximum(sol1, sol2)


def mc_estimates(run_sum, run_sum_squares, n):
    """Returns sample mean and variance from sum and sum of squares

    :param run_sum: float
        The sum of the samples

    :param run_sum_squares: float
        The sum of the squares of the samples

    :param n: int
        The number of samples

    :return: 2-tuple of floats
         The sample mean and the sample variance
    """
    sample_mean = run_sum / n
    sample_var = ((run_sum_squares - (run_sum * run_sum) / n) / (n - 1))
    return sample_mean, sample_var


def remove_steps(time_tol, steps, time_interval):
    """Returns index of last step when a tolerance is removed from the interval"""
    steps_to_remove = time_tol / (time_interval / steps)
    return int(np.floor(steps - steps_to_remove))


def get_corr_matrix(rhos):
    """Returns correlation matrix when given a vector of correlations

    :param rhos: torch.tensor
        The correlations

    :return: torch.tensor
        The correlation matrix
    """
    rhos_tensor = torch.tensor(rhos)
    t_n = len(rhos)
    sol = solve_quadratic((torch.tensor(1), torch.tensor(1), torch.tensor(-2 * t_n)))
    assert torch.isclose(sol, sol.floor()), "Length of correlation vector is not triangular"
    n = int(sol + 1)

    corr_matrix = torch.eye(n)
    triu_indices = torch.triu_indices(row=n, col=n, offset=1)
    corr_matrix[triu_indices[0], triu_indices[1]] = rhos_tensor
    corr_matrix += torch.triu(corr_matrix, 1).t()
    try:
        torch.linalg.cholesky(corr_matrix)
    except:
        raise RuntimeError('Matrix is not positive semidefinite')
    return corr_matrix


def ceil_mult(x, n):
    """Returns smallest multiple of n greater than or equal to x

    :param x: float

    :param n: int

    :return: int
    """
    factor = float(x) / n
    return int(np.ceil(factor) * n)


def get_jump_comp(c_plus, c_minus, alpha, mu, f):
    def g(x):
        return np.exp(f * x) - 1 - f * x

    def q(x):
        return np.exp(f * x) - 1

    def tail(x):
        return np.exp(-mu * (np.abs(x) - 1))

    def inner(x):
        return np.abs(x) ** (- alpha - 1)

    i1 = quad(lambda x: c_minus * q(x) * tail(x), -np.inf, -1)[0]
    i2 = quad(lambda x: c_minus * g(x) * inner(x), -1, 0)[0]
    i3 = quad(lambda x: c_plus * g(x) * inner(x), 0, 1)[0]
    i4 = quad(lambda x: c_plus * q(x) * tail(x), 1, np.inf)[0]
    return i1 + i2 + i3 + i4
