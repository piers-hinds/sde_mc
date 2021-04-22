import torch
import matplotlib.pyplot as plt


def basis_1(x):
    return torch.ones(x.shape[0], dtype=torch.float64)


def basis_2(x):
    return x[:, 0]


def basis_3(x):
    return x[:, 0]**2


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