import torch
import numpy as np
from abc import abstractmethod
from .sde import Sde
from .helpers import get_jump_comp

UNIFORM_TOL = 5.960464477539063e-08


class InverseCdf:
    def __init__(self, c_minus, c_plus, mu, alpha, epsilon):
        self.cm = c_minus
        self.cp = c_plus
        self.mu = mu
        self.alpha = alpha
        self.eps = epsilon
        self.lda = (c_minus + c_plus) * (1 / mu + (epsilon ** (-alpha) - 1) / alpha)
        self.y1 = (self.cm / (self.mu * self.lda))
        self.y2 = (1 / self.lda) * ((self.cm / self.mu) + (self.cm * (self.eps ** (-self.alpha) - 1) / self.alpha))
        self.y3 = 1 - self.cp / (self.mu * self.lda)

    def __call__(self, y):
        x1 = torch.log((self.mu * self.lda * y) / self.cm) / self.mu - 1
        x2 = - (self.alpha * ((self.lda * y / self.cm) - (1 / self.mu)) + 1) ** (-1 / self.alpha)
        x3 = ((-self.alpha / self.cp) * (self.lda * y - self.cm / self.mu - self.cm * (
                    (self.eps ** (-self.alpha) - 1) / self.alpha)) + self.eps ** (-self.alpha)) ** (-1 / self.alpha)
        x4 = 1 - (1 / self.mu) * torch.log(self.mu * self.lda * (1 - y) / self.cp)
        return torch.where(y <= self.y1, x1,
                           torch.where(y < self.y2, x2,
                                       torch.where(y < self.y3, x3, x4)))


class Levy:
    """ A Levy-driven SDE with infinite activity"""

    def __init__(self, dim, icdf):
        self.dim = dim
        self.icdf = icdf

    @abstractmethod
    def drift(self, t, x):
        pass

    @abstractmethod
    def diffusion(self, t, x):
        pass

    @abstractmethod
    def jumps(self, t, x, jumps):
        pass

    @abstractmethod
    def gamma(self):
        pass

    @abstractmethod
    def beta(self):
        pass

    @abstractmethod
    def jump_mean(self):
        pass
    

class LevySde(Sde):
    """An Sde interface for a Levy-driven SDE which approximates small jumps with an additional diffusion
    component"""

    def __init__(self, levy, init_value, corr_matrix=None, scale_jump_rate=False, device='cpu', seed=1):
        super(LevySde, self).__init__(init_value, levy.dim, levy.dim * 2, 'indep', corr_matrix)
        self.levy = levy
        self.scale_rate = scale_jump_rate

    def drift(self, t, x):
        return self.levy.drift(t, x) - self.levy.jumps(t, x, 1) * self.levy.gamma()

    def diffusion(self, t, x):
        """Needs to combine both diffusions into a vector. Need to add some noise_type parameter
        for when one dimension depends on multiple Brownian motions. Or noise_dim != dim"""
        return torch.stack([self.levy.diffusion(t, x), self.levy.jumps(t, x, 1) * self.levy.beta()], dim=-1)

    def jumps(self, t, x, jumps):
        return self.levy.jumps(t, x, jumps)

    def sample_jumps(self, size, device):
        unifs = torch.rand(size, device=device) + UNIFORM_TOL / 3
        return self.levy.icdf(unifs)

    def jump_rate(self):
        if self.scale_rate:
            return torch.tensor(self.levy.icdf.lda * self.levy.dim)
        else:
            return torch.tensor(self.levy.icdf.lda)

    def jump_mean(self):
        return self.levy.jump_mean()


class ExampleLevy(Levy):
    def __init__(self, c_plus, c_minus, alpha, mu, r, sigma, f, chol_corr, epsilon, dim):
        super(ExampleLevy, self).__init__(dim, InverseCdf(c_minus, c_plus, mu, alpha, epsilon))
        self.cm = c_minus
        self.cp = c_plus
        self.alpha = alpha
        self.mu = mu
        self.f = f
        self.sigma = sigma
        self.epsilon = epsilon
        self.jump_comp = torch.tensor([get_jump_comp(c_plus, c_minus, alpha, mu, f[i].item()) for i in range(dim)], device=f.device)
        self.sigma_matrix = torch.matmul(torch.diag(sigma), chol_corr)
        self.row_sum_sq = (self.sigma_matrix**2).sum(-1)

    def drift(self, t, x):
        return - 0.5 * self.row_sum_sq - self.jump_comp

    def diffusion(self, t, x):
        return self.sigma * torch.ones_like(x)

    def jumps(self, t, x, jumps):
        return self.f * jumps * torch.ones_like(x)

    def gamma(self):
        return (self.cp - self.cm) * (1 - self.epsilon ** (1 - self.alpha)) / (1 - self.alpha)

    def beta(self):
        return np.sqrt((self.cp + self.cm) * (self.epsilon ** (2 - self.alpha)) / (2 - self.alpha))

    def jump_mean(self):
        return 0


class ExpExampleLevy(Levy):
    def __init__(self, c_minus, c_plus, alpha, mu, r, sigma, f, epsilon, dim=1):
        super(ExpExampleLevy, self).__init__(dim, InverseCdf(c_minus, c_plus, mu, alpha, epsilon))
        self.cm = c_minus
        self.cp = c_plus
        self.alpha = alpha
        self.mu = mu
        self.r = r
        self.sigma = sigma
        self.f = f
        self.epsilon = epsilon

    def drift(self, t, x):
        return self.r * x

    def diffusion(self, t, x):
        return x * self.sigma

    def jumps(self, t, x, jumps):
        return self.f * x * jumps

    def gamma(self):
        return (self.cp - self.cm) * (1 - self.epsilon ** (1 - self.alpha)) / (1 - self.alpha)

    def beta(self):
        return np.sqrt((self.cp + self.cm) * (self.epsilon ** (2 - self.alpha)) / (2 - self.alpha))
    
    def jump_mean(self):
        return 0


class Levy2d(Levy):
    def __init__(self, c_plus, c_minus, alpha, mu, f, epsilon):
        super(Levy2d, self).__init__(2, InverseCdf(c_minus, c_plus, mu, alpha, epsilon))
        self.cp = c_plus
        self.cm = c_minus
        self.alpha = alpha
        self.mu = mu
        self.f = f
        self.epsilon = epsilon

    def drift(self, t, x):
        return torch.ones_like(x) * (- self.f * (self.cp - self.cm) * (1 / self.mu + 1 / (self.mu**2)))

    def diffusion(self, t, x):
        return torch.ones_like(x)

    def jumps(self, t, x, jumps):
        return self.f * jumps * torch.ones_like(x)

    def gamma(self):
        return (self.cp - self.cm) * (1 - self.epsilon**(1 - self.alpha)) / (1- self.alpha)

    def beta(self):
        return np.sqrt((self.cp + self.cm) * (self.epsilon**(2 - self.alpha)) / (2 - self.alpha))

    def jump_mean(self):
        if self.cp == self.cm:
            return 0
        measure = (self.cp + self.cm) * (1 / self.mu + (self.epsilon**(-self.alpha) - 1) / (self.alpha))
        return (self.cp - self.cm) * ( (1 / self.mu + 1 / (self.mu**2)) +  ((1 - self.epsilon**(1-self.alpha)) / (1 - self.alpha)) ) / measure
