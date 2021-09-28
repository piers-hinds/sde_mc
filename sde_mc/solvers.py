import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from .helpers import solve_quadratic


class SdeSolver:
    """A class for generating trajectories of SDEs: contains an SDE object and information about the numerical scheme"""

    def __init__(self, sde, time_interval, num_steps, device='cpu', seed=1):
        """
        :param sde: Sde
            The SDE to solve

        :param time_interval: float
            The time to solve up to

        :param num_steps: int
            The number of steps in the discretization

        :param device: string
            The device to do the computations on (default 'cpu')

        :param seed: int
            Seed for torch (default 1)
        """
        self.sde = sde
        self.time_interval = time_interval
        self.num_steps = num_steps
        self.device = device
        self.has_jumps = self.sde.jump_rate().any()

        if len(self.sde.corr_matrix) > 1:
            self.lower_cholesky = torch.linalg.cholesky(self.sde.corr_matrix.to(device))
        else:
            self.lower_cholesky = torch.tensor([[1.]], device=device)
        torch.manual_seed(seed)

    def solve(self, bs=1, return_normals=False, method=None):
        """Solves the SDE using the method specified

        :param bs: int (default = 1)
            The batch size (the number of paths simulated simultaneously)

        :param return_normals: bool (default = False)
            If True returns the normal random variables used

        :param method: string (default = None)
            The numerical method to use

        :return: torch.tensor of shape (bs, steps, dimensions)
            The paths simulated across
        """
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

        :param bs: int
            The batch size (the number of paths simulated simultaneously)

        :param return_normals: bool
            If True returns the normal random variables used

        :return: torch.tensor of shape (bs, steps, dimensions)
            The paths simulated across
        """
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

        return paths, (corr_normals, jumps)
        
    def heston(self, bs=1, return_normals=False):
        """Custom scheme specifically for the Heston model. The asset price is simulated by the explicit
        Euler scheme, while the variance is simulated using the fully implicit Euler scheme to preserve 
        positivity."""
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
                self.sde.diffusion(t, paths_coarse[:, i]) * \
                torch.sum(corr_normals[:, (i * factor):((i + 1) * factor)], dim=1)
        return (paths_fine, paths_coarse), (corr_normals, None)


class JumpAdaptedSolver(SdeSolver):
    def __init__(self, sde, time_interval, num_steps, device='cpu', seed=1):
        super(JumpAdaptedSolver, self).__init__(sde, time_interval, num_steps, device, seed)
        self.MAX_JUMPS = max(int(self.time_interval * self.sde.jump_rate().sum() * 10), 5)

    def euler(self, bs, return_normals=False):
        bs = int(bs)
        h = torch.tensor(self.time_interval / self.num_steps, device=self.device)

        # at most there will be num_steps + MAX_JUMPS + 1 number of observations
        paths = torch.zeros(size=(bs, self.num_steps + self.MAX_JUMPS + 1, self.sde.dim), device=self.device)
        left_paths = torch.zeros_like(paths)
        time_paths = torch.zeros(size=(bs, self.num_steps + self.MAX_JUMPS + 1, 1), device=self.device) + self.time_interval
        x = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        paths[:, 0] = x
        left_paths[:, 0] = x
        t = torch.zeros((bs, 1), device=self.device)
        time_paths[:, 0] = t

        # storage for normals
        normals = torch.zeros(size=(bs, self.num_steps + self.MAX_JUMPS, self.sde.dim), device=self.device)
        # storage for jump paths
        jump_paths = torch.zeros_like(paths)

        jump_times = self.sample_jump_times(size=(bs, self.MAX_JUMPS, 1))
        jump_idxs = torch.zeros_like(jump_times[:, 0, :]).long()

        total_steps = 0
        while torch.any(t < self.time_interval):
            # counts the number of steps in total
            total_steps += 1

            # times and sizes of the next jump
            next_jump_time = jump_times[torch.arange(bs), jump_idxs.squeeze(-1), :]

            # time step is minimum of (mesh size, time to next jump, time to end of interval)
            h = torch.minimum(h, torch.maximum(self.time_interval - t, torch.tensor(0.)))
            dt = torch.minimum(h, next_jump_time - t)
            assert (next_jump_time > t).all()
            # step diffusion until the next time step
            x, normals[:, total_steps - 1] = self.step_diffusion(t, x, dt)
            left_paths[:, total_steps] = x
            t += dt
            time_paths[:, total_steps] = t

            # add jumps if the next jump is now - could sample jumps here if storage issues
            next_jump_size = self.sample_one_jump(bs)
            current_jumps = torch.where(torch.isclose(next_jump_time, t), next_jump_size,
                                        torch.zeros_like(next_jump_size))
            jump_paths[:, total_steps] = current_jumps
            x += self.add_jumps(t, x, current_jumps)

            # store in path
            paths[:, total_steps] = x

            # increment jump index if a jump has just happened
            jump_idxs = torch.where(torch.isclose(next_jump_time, t), jump_idxs + 1, jump_idxs)

        return paths, (normals, time_paths, left_paths, total_steps, jump_paths)

    def step_diffusion(self, t, x, h):
        corr_normals = self.sample_corr_normals(x.shape + torch.Size([1]), h.unsqueeze(-1))
        return x + self.sde.drift(t, x) * h + self.sde.diffusion(t, x) * corr_normals, corr_normals

    def add_jumps(self, t, x, jumps):
        return self.sde.jumps(t, x, jumps)

    def sample_jump_times(self, size):
        return torch.empty(size, device=self.device).exponential_(self.sde.jump_rate().sum()).cumsum(dim=1)

    def sample_one_jump(self, size):
        jumps = self.sde.sample_jumps([size, self.sde.dim], self.device)
        # Following line can be changed when rates are not all equal, use torch.Categorical
        rand_dim = torch.randint(0, self.sde.dim, (size,), device=self.device)
        return F.one_hot(rand_dim, num_classes=self.sde.dim) * jumps


class Grid(ABC):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.time_interval = end-start
        self.t = start

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass


class UniformGrid(Grid):
    def __init__(self, start, end, num_steps):
        super(UniformGrid, self).__init__(start, end)
        self.num_steps = num_steps
        self.h = (end-start) / num_steps
        assert self.h > 1e-8

    def __next__(self):
        if self.t > self.end - 1e-8:
            raise StopIteration
        t = self.t
        self.t += self.h
        return t
