import torch
from .helpers import solve_quadratic


class SdeSolver:
    """A class for generating trajectories of SDEs: contains an SDE object and information about the numerical scheme"""

    def __init__(self, sde, time_interval, num_steps, device='cpu', seed=1):
        """
        :param sde: Sde, the SDE to solve
        :param time_interval: float, the time to solve up to
        :param num_steps: int, the number of steps in the discretization
        :param device: string, the device to do the computations on (default 'cpu')
        :param seed: int, seed for torch (default 1)
        """
        self.sde = sde
        self.time_interval = time_interval
        self.num_steps = num_steps
        self.device = device
        self.has_jumps = bool(self.sde.jump_rate())

        if len(self.sde.corr_matrix) > 1:
            self.lower_cholesky = torch.linalg.cholesky(self.sde.corr_matrix.to(device))
        else:
            self.lower_cholesky = torch.tensor([[1.]], device=device)
        torch.manual_seed(seed)

    def solve(self, bs=1, return_normals=False, method=None):
        if method is None:
            method = self.sde.simulation_method
        if method == 'euler':
            return self.euler(bs, return_normals)
        if method == 'heston':
            return self.heston(bs, return_normals)
    
    def sample_corr_normals(self, size, h):
        normals = torch.randn(size=size, device=self.device) * torch.sqrt(h)
        return torch.matmul(self.lower_cholesky, normals).squeeze(-1)

    def sample_bernoullis(self, size, h):
        rates = torch.ones(size=size, device=self.device) * (self.sde.jump_rate() * h)
        return torch.bernoulli(rates)

    def sample_jumps(self, size):
        return self.sde.sample_jumps(size, self.device)

    def euler(self, bs, return_normals=False):
        """Implements the Euler method for solving SDEs
        :param bs: int, the batch size (the number of paths simulated simultaneously)
        :param return_normals: bool, if True returns the normal random variables used
        :return: torch.tensor, the paths simulated across (bs, steps, dimensions)
        """
        assert bs >= 1, "Batch size must at least one"
        bs = int(bs)

        h = torch.tensor(self.time_interval / self.num_steps, device=self.device)

        paths = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)

        paths[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

        corr_normals = self.sample_corr_normals(size=(bs, self.num_steps, self.sde.dim, 1), h=h)

        t = torch.tensor(0.0, device=self.device)

        if self.has_jumps:
            bernoullis = self.sample_bernoullis(size=(bs, self.num_steps, self.sde.dim), h=h)
            max_jumps = self.sample_jumps(size=(bs, self.num_steps, self.sde.dim))
            jumps = (max_jumps * bernoullis)
        else:
            jumps = None

        for i in range(self.num_steps):
            paths[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * h + \
                                       self.sde.diffusion(t, paths[:, i]) * corr_normals[:, i]
            if self.has_jumps:
                paths[:, i + 1] += self.sde.jumps(t, paths[:, i], jumps[:, i])
            t += h
        additional_data = (corr_normals, jumps)
        # else:
        #     for i in range(self.num_steps):
        #         paths[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * h + \
        #                           self.sde.diffusion(t, paths[:, i]) * corr_normals[:, i]
        #         t += h
        #     additional_data = (corr_normals, None)

        if return_normals:
            return paths, additional_data
        else:
            return paths, None
        
    def heston(self, bs=1, return_normals=False):
        """Custom scheme specifically for the Heston model. The asset price is simulated by the explicit
        Euler scheme, while the variance is simulated using the fully implicit Euler scheme to preserve 
        positivity."""
        assert bs >= 1, "Batch size must at least one"
        bs = int(bs)

        h = torch.tensor(self.time_interval / self.num_steps, device=self.device)

        paths = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)

        paths[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

        corr_normals = self.sample_corr_normals(size=(bs, self.num_steps, self.sde.dim, 1), h=h)

        t = torch.tensor(0.0, device=self.device)

        for i in range(self.num_steps):
                paths[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * h + \
                                  self.sde.diffusion(t, paths[:, i]) * corr_normals[:, i]
                coefs = self.sde.quadratic_parameters(paths[:, i, 1], h, corr_normals[:, i, 1])
                sol = solve_quadratic(coefs)
                paths[:, i + 1, 1] = sol * sol
                t += h
        return paths, (corr_normals, None)

    def multilevel_euler(self, bs, levels, return_normals=False):
        bs = int(bs)
        fine, coarse = levels
        factor = int(fine / coarse)

        h = torch.tensor(self.time_interval / fine, device=self.device)
        paths_fine = torch.empty(size=(bs, fine + 1, self.sde.dim), device=self.device)
        paths_coarse = torch.empty(size=(bs, coarse + 1, self.sde.dim), device=self.device)
        paths_fine[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        paths_coarse[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

        corr_normals = self.sample_corr_normals((bs, fine, self.sde.dim, 1), h=h)
        t = torch.tensor(0.0, device=self.device)

        for i in range(coarse):
            for j in range(factor):
                # step the fine approximation
                paths_fine[:, i * factor + j + 1] = paths_fine[:, i * factor + j] + \
                                                    self.sde.drift(t, paths_fine[:, i * factor + j]) * h + \
                                                    self.sde.diffusion(t, paths_fine[:, i * factor + j]) * \
                                                    corr_normals[:, i * factor + j]
            # step the coarse approximation
            paths_coarse[:, i + 1] = paths_coarse[:, i] + self.sde.drift(t, paths_coarse[:, i]) * h * factor + \
                                     self.sde.diffusion(t, paths_coarse[:, i]) * torch.sum(
                corr_normals[:, (i * factor):((i + 1) * factor)], dim=1)
        return (paths_fine, paths_coarse), (None, corr_normals, None)
