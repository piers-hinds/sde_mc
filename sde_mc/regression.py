import torch
import numpy as np
import matplotlib.pyplot as plt


def basis_1(x, t):
    alpha = 10 * t / 3 + 0.01 * (3 - t) / 3
    return (torch.atan(alpha * (x - 1)) + torch.atan(alpha)) / np.pi


def basis_2(x, t):
    beta = 0.0001 * t / 3 + 0.005 * (3 - t) / 3
    return (x / 2) + x * (x - 2) / (4 * (torch.sqrt((x - 1)**2 / 4 + beta) + torch.sqrt(0.25 + beta)))


def basis_3(x, t):
    return x / (8 + x**2)


def d_basis_1(x, t):
    x.requires_grad = True
    y = basis_1(x, t)
    y.backward(torch.ones_like(x))
    x.requires_grad = False
    return x.grad


def d_basis_2(x, t):
    x.requires_grad = True
    y = basis_2(x, t)
    y.backward(torch.ones_like(x))
    x.requires_grad = False
    return x.grad


def d_basis_3(x, t):
    x.requires_grad = True
    y = basis_3(x, t)
    y.backward(torch.ones_like(x))
    x.requires_grad = False
    return x.grad


def fit_basis(x, y, basis):
    """
    Fits basis functions to the data (x, y) by minimizing MSE
    :param x: torch.tensor,
    :param y: torch.tensor,
    :param basis: list of functions, 
    :return: torch.tensor, coefficients of basis functions
    """
    assert x.shape[0] == y.shape[0]
    num_basis = len(basis)
    num_obs = x.shape[0]

    a = torch.empty((num_basis, num_basis))
    b = torch.empty(num_basis)

    for i in range(num_basis):
        for j in range(num_basis):
            a[i, j] = torch.dot(basis[i](x), basis[j](x)) / num_obs
        b[i] = torch.dot(basis[i](x), y) / num_obs
    output = torch.matmul(torch.linalg.inv(a), b)
    return output


def plot_basis(coefs, basis, x_min, x_max):
    """
    Plots the linear combination of basis functions onto an existing plot
    :param coefs: torch.tensor, the coefficients of the basis functions
    :param basis: list, a list of the basis functions
    :param x_min: torch.tensor, the minimum x value
    :param x_max: torch.tensor, the maximum x value
    :return:
    """
    x_line = torch.linspace(x_min, x_max, 100).unsqueeze(-1)
    y_line = 0
    for i in range(len(basis)):
        y_line += coefs[i] * basis[i](x_line)
    plt.plot(x_line, y_line, color='green')