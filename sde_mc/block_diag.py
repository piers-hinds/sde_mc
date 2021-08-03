"""A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk"""
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


def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3

    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


def solve_quadratic(coefficients):
    """Solves quadratics across batches, returns maxmimum root"""
    a, b, c = coefficients
    sol1 = (- b + torch.sqrt(b * b - 4 * a * c) ) / (2 * a)
    sol2 = (- b - torch.sqrt(b * b - 4 * a * c) ) / (2 * a)
    return torch.maximum(sol1, sol2)
