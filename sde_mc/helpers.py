import torch


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
