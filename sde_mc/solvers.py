import torch
import torch.nn.functional as F
from scipy.stats import poisson
from abc import ABC, abstractmethod
from .schemes import EulerScheme, HestonScheme
from .helpers import solve_quadratic


class SdeSolver(ABC):
    def __init__(self, sde, time_interval, num_steps, device='cpu', seed=1):
        """
        :param sde: Sde
            The SDE to solve

        :param time_interval: float
            The time to solve up to

        :param num_steps: int
            The number of steps in the discretisation

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

    @abstractmethod
    def solve(self, bs=1, return_normals=False):
        pass

    @abstractmethod
    def init_storage(self, bs, steps):
        pass

    @abstractmethod
    def step(self, t, x, h, corr_normals):
        pass

    def sample_corr_normals(self, size, h, corr=True):
        normals = torch.randn(size=size, device=self.device) * torch.sqrt(h)
        if corr:
            return torch.matmul(self.lower_cholesky, normals).squeeze(-1)
        else:
            return normals.squeeze(-1)


class DiffusionSolver(SdeSolver):
    @abstractmethod
    def step(self, t, x, h, corr_normals):
        pass

    def init_storage(self, bs, steps):
        paths = torch.empty(size=(bs, steps + 1, self.sde.dim), device=self.device)
        return paths

    def solve(self, bs=1, return_normals=False):
        bs = int(bs)
        h = torch.tensor(self.time_interval / self.num_steps, device=self.device)
        x = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        t = torch.tensor(0.0, device=self.device)

        paths = self.init_storage(bs, self.num_steps)
        paths[:, 0] = x

        if self.sde.diffusion_struct == 'diag':
            corr_normals = self.sample_corr_normals(size=(bs, self.num_steps, self.sde.dim, 1), h=h)
        if self.sde.diffusion_struct == 'indep':
            corr_normals = self.sample_corr_normals(size=(bs, self.num_steps, self.sde.dim, int(self.sde.brown_dim /
                                                          self.sde.dim)), h=h)

        for i in range(self.num_steps):
            x = self.step(t, x, h, corr_normals[:, i])
            if True:
                paths[:, i + 1] = x
            t += h
        return paths, corr_normals


class EulerSolver(EulerScheme, DiffusionSolver):
    pass


class HestonSolver(HestonScheme, DiffusionSolver):
    pass


class JumpDiffusionSolver(SdeSolver):
    def __init__(self, sde, time_interval, num_steps, device='cpu', seed=1):
        super(JumpDiffusionSolver, self).__init__(sde, time_interval, num_steps, device, seed)
        self.max_jumps = max(int(self.time_interval * poisson.ppf(1 - 1/1e9, self.sde.jump_rate().sum())), 5)

    @abstractmethod
    def step(self, t, x, h, corr_normals):
        pass

    def add_jumps(self, t, x, jumps):
        return x + self.sde.jumps(t, x, jumps)

    def sample_jump_times(self, size):
        return torch.empty(size, device=self.device).exponential_(self.sde.jump_rate().sum()).cumsum(dim=1)

    def sample_one_jump(self, size):
        jumps = self.sde.sample_jumps([size, 1], self.device).repeat(1, self.sde.dim)
        return jumps

    def init_storage(self, bs, steps):
        paths = torch.zeros(size=(bs, steps + 1, self.sde.dim), device=self.device)
        left_paths = torch.zeros_like(paths)
        jump_paths = torch.zeros_like(paths)
        time_paths = torch.zeros(size=(bs, steps + 1, 1), device=self.device) + self.time_interval
        if self.sde.diffusion_struct == 'diag':
            normals = torch.zeros(size=(bs, steps, self.sde.dim), device=self.device)
        else:
            normals = torch.zeros(size=(bs, steps, self.sde.dim, int(self.sde.brown_dim / self.sde.dim)), device=self.device)
        return paths, left_paths, time_paths, jump_paths, normals

    def solve(self, bs=1, return_normals=False):
        bs = int(bs)
        h = torch.tensor(self.time_interval / self.num_steps, device=self.device)
        x = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        t = torch.zeros((bs, 1), device=self.device)

        paths, left_paths, time_paths, jump_paths, normals = self.init_storage(bs, self.num_steps + self.max_jumps)
        paths[:, 0] = x
        left_paths[:, 0] = x
        time_paths[:, 0] = t

        jump_times = self.sample_jump_times(size=(bs, self.max_jumps, 1))
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
            assert (next_jump_time >= t).all()
            # step diffusion until the next time step
            if self.sde.diffusion_struct == 'diag':
                corr_normals = self.sample_corr_normals(x.shape + torch.Size([1]), h.unsqueeze(-1))
            else:
                corr_normals = torch.stack([
                                            self.sample_corr_normals(x.shape + torch.Size([1]), h.unsqueeze(-1)),
                                            self.sample_corr_normals([x.shape[0], 1, 1], h.unsqueeze(-1), corr=False).repeat(1, x.shape[1])
                                            ], dim=-1)
            x = self.step(t, x, dt, corr_normals)
            normals[:, total_steps - 1] = corr_normals
            left_paths[:, total_steps] = x
            t += dt
            time_paths[:, total_steps] = t

            # add jumps if the next jump is now - could sample jumps here if storage issues
            next_jump_size = self.sample_one_jump(bs)
            current_jumps = torch.where(torch.isclose(next_jump_time, t, atol=1e-12), next_jump_size,
                                        torch.zeros_like(next_jump_size))
            jump_paths[:, total_steps] = current_jumps
            x = self.add_jumps(t, x, current_jumps)

            # store in path
            paths[:, total_steps] = x

            # increment jump index if a jump has just happened
            jump_idxs = torch.where(torch.isclose(next_jump_time, t, atol=1e-12), jump_idxs + 1, jump_idxs)
        return paths, (normals, time_paths, left_paths, total_steps, jump_paths)


class JumpEulerSolver(EulerScheme, JumpDiffusionSolver):
    pass


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
