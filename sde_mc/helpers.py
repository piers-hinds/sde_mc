import torch


def partition(interval, steps, ends='right', device='cpu'):
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
    """Solves quadratics across batches, returns maximum root"""
    a, b, c = coefficients
    sol1 = (- b + torch.sqrt(b * b - 4 * a * c)) / (2 * a)
    sol2 = (- b - torch.sqrt(b * b - 4 * a * c)) / (2 * a)
    return torch.maximum(sol1, sol2)
