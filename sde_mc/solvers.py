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

    def multilevel_solve(self, bs, levels, return_normals=False):
        bs = int(bs)
        fine, coarse = levels
        factor = int(fine / coarse)

        h_fine = torch.tensor(self.time_interval / fine, device=self.device)
        h_coarse = factor * h_fine
        t = torch.tensor(0.0, device=self.device)
        x_fine = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        x_coarse = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        paths_fine = self.init_storage(bs, fine)
        paths_coarse = self.init_storage(bs, coarse)
        paths_fine[:, 0] = x_fine
        paths_coarse[:, 0] = x_coarse

        if self.sde.diffusion_struct == 'diag':
            corr_normals = self.sample_corr_normals(size=(bs, fine, self.sde.dim, 1), h=h_fine)
        if self.sde.diffusion_struct == 'indep':
            corr_normals = self.sample_corr_normals(size=(bs, fine, self.sde.dim, int(self.sde.brown_dim /
                                                                                      self.sde.dim)), h=h_fine)

        for i in range(coarse):
            for j in range(factor):
                x_fine = self.step(t, x_fine, h_fine, corr_normals[:, i * factor + j])
                t += h_fine
                paths_fine[:, i * factor + j + 1] = x_fine
            x_coarse = self.step(t, x_coarse, h_coarse,
                                 torch.sum(corr_normals[:, (i * factor):((i + 1) * factor)], dim=1))
            paths_coarse[:, i + 1] = x_coarse
        return (paths_fine, paths_coarse), corr_normals


class EulerSolver(EulerScheme, DiffusionSolver):
    pass


class HestonSolver(HestonScheme, DiffusionSolver):
    pass


class JumpDiffusionSolver(SdeSolver):
    def __init__(self, sde, time_interval, num_steps, device='cpu', seed=1, exact_jumps=False):
        super(JumpDiffusionSolver, self).__init__(sde, time_interval, num_steps, device, seed)
        self.max_jumps = max(int(self.time_interval * poisson.ppf(1 - 1 / 1e9, self.sde.jump_rate().sum())), 5)
        self.exact_jumps = exact_jumps

    @abstractmethod
    def step(self, t, x, h, corr_normals):
        pass

    def add_jumps(self, t, old_x, x, jumps):
        return x + self.sde.jumps(t, old_x, jumps)

    def sample_jump_times(self, size):
        return torch.empty(size, device=self.device).exponential_(self.sde.jump_rate().sum()).cumsum(dim=1)

    def sample_one_jump(self, size):
        jumps = self.sde.sample_jumps([size, 1], self.device).repeat(1, self.sde.dim)
        return jumps

    def init_storage(self, bs, steps, low_storage=False):
        paths = torch.zeros(size=(bs, steps + 1, self.sde.dim), device=self.device)
        if low_storage:
            return paths, None, None, None, None
        left_paths = torch.zeros_like(paths)
        jump_paths = torch.zeros_like(paths)
        time_paths = torch.zeros(size=(bs, steps + 1, 1), device=self.device) + self.time_interval
        if self.sde.diffusion_struct == 'diag':
            normals = torch.zeros(size=(bs, steps, self.sde.dim), device=self.device)
        else:
            normals = torch.zeros(size=(bs, steps, self.sde.dim, int(self.sde.brown_dim / self.sde.dim)),
                                  device=self.device)
        return paths, left_paths, time_paths, jump_paths, normals

    def solve(self, bs=1, return_normals=False, low_storage=False):
        bs = int(bs)
        h = torch.tensor(self.time_interval / self.num_steps, device=self.device)
        x = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        t = torch.zeros((bs, 1), device=self.device)

        paths, left_paths, time_paths, jump_paths, normals = self.init_storage(bs, self.num_steps + self.max_jumps,
                                                                               low_storage)
        paths[:, 0] = x

        if not low_storage:
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

            # time step is minimum of (prescribed maximum mesh size, time to next jump, time to end of interval)
            h = torch.minimum(h, torch.maximum(self.time_interval - t, torch.tensor(0.)))
            dt = torch.minimum(h, next_jump_time - t)

            assert (next_jump_time >= t).all()
            # step diffusion until the next time step
            if self.sde.diffusion_struct == 'diag':
                corr_normals = self.sample_corr_normals(x.shape + torch.Size([1]), dt.unsqueeze(-1))
            else:
                corr_normals = torch.stack([
                    self.sample_corr_normals(x.shape + torch.Size([1]), dt.unsqueeze(-1)),
                    self.sample_corr_normals([x.shape[0], 1, 1], dt.unsqueeze(-1), corr=False).repeat(1, x.shape[1])
                ], dim=-1)
            old_x = x
            x = self.step(t, x, dt, corr_normals)
            t += dt
            if not low_storage:
                normals[:, total_steps - 1] = corr_normals
                left_paths[:, total_steps] = x
                time_paths[:, total_steps] = t

            # add jumps if the next jump is now
            next_jump_size = self.sample_one_jump(bs)
            current_jumps = torch.where(torch.isclose(next_jump_time, t, atol=1e-12), next_jump_size,
                                        torch.zeros_like(next_jump_size))
            if self.exact_jumps:
                x = self.add_jumps(t, x, x, current_jumps)
            else:
                x = self.add_jumps(t, old_x, x, current_jumps)

            # store in path
            paths[:, total_steps] = x
            if not low_storage:
                jump_paths[:, total_steps] = current_jumps

            # increment jump index if a jump has just happened
            jump_idxs = torch.where(torch.isclose(next_jump_time, t, atol=1e-12), jump_idxs + 1, jump_idxs)
        return paths[:, :total_steps + 1], (normals, time_paths, left_paths, total_steps, jump_paths)

    def multilevel_solve(self, bs, levels, return_normals=False):
        bs = int(bs)
        fine, coarse = levels
        factor = int(fine / coarse)

        h_fine = torch.tensor(self.time_interval / fine, device=self.device)
        h_coarse = factor * h_fine

        t_fine = torch.zeros((bs, 1), device=self.device)
        t_coarse = torch.zeros((bs, 1), device=self.device)

        x_fine = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        x_coarse = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

        paths_fine, _, _, _, _ = self.init_storage(bs, coarse + self.max_jumps, low_storage=True)
        paths_coarse, _, _, _, _ = self.init_storage(bs, coarse + self.max_jumps, low_storage=True)
        paths_fine[:, 0] = x_fine
        paths_coarse[:, 0] = x_coarse

        jump_times = self.sample_jump_times(size=(bs, self.max_jumps, 1))
        jump_idxs = torch.zeros_like(jump_times[:, 0, :]).long()

        total_steps_fine = 0
        total_steps_coarse = 0
        while torch.any(t_fine < self.time_interval):
            run_sum_normals = 0

            # times and sizes of the next jump
            next_jump_time = jump_times[torch.arange(bs), jump_idxs.squeeze(-1), :]

            # Do factor steps for the finer level - store the normals
            for i in range(factor):
                total_steps_fine += 1
                # time step is minimum of (mesh size, time to next jump, time to end of interval)
                h_fine = torch.minimum(h_fine, torch.maximum(self.time_interval - t_fine, torch.tensor(0.)))
                dt_fine = torch.minimum(h_fine, next_jump_time - t_fine)
                assert (next_jump_time >= t_fine).all()
                # step diffusion until the next time step
                if self.sde.diffusion_struct == 'diag':
                    corr_normals = self.sample_corr_normals(x_fine.shape + torch.Size([1]), dt_fine.unsqueeze(-1))
                else:
                    corr_normals = torch.stack([
                        self.sample_corr_normals(x_fine.shape + torch.Size([1]), dt_fine.unsqueeze(-1)),
                        self.sample_corr_normals([x_fine.shape[0], 1, 1], dt_fine.unsqueeze(-1), corr=False).repeat(1,
                                                                                                                    x_fine.shape[
                                                                                                                        1])
                    ], dim=-1)
                old_x_fine = x_fine
                x_fine = self.step(t_fine, x_fine, dt_fine, corr_normals)
                t_fine += dt_fine
                run_sum_normals += corr_normals

            # Do one step on the coarser level
            total_steps_coarse += 1
            h_coarse = torch.minimum(h_coarse, torch.maximum(self.time_interval - t_coarse, torch.tensor(0.)))
            dt_coarse = torch.minimum(h_coarse, next_jump_time - t_coarse)
            old_x_coarse = x_coarse
            x_coarse = self.step(t_coarse, x_coarse, dt_coarse, run_sum_normals)  # check this
            t_coarse += dt_coarse

            # add jumps if the next jump is now - could sample jumps here if storage issues
            assert torch.isclose(t_fine, t_coarse, atol=1e-12).all()
            next_jump_size = self.sample_one_jump(bs)
            current_jumps = torch.where(torch.isclose(next_jump_time, t_fine, atol=1e-12), next_jump_size,
                                        torch.zeros_like(next_jump_size))
            # jump_paths_fine[:, total_steps_fine] = current_jumps
            if self.exact_jumps:
                x_fine = self.add_jumps(t_fine, x_fine, x_fine, current_jumps)
                x_coarse = self.add_jumps(t_coarse, x_coarse, x_coarse, current_jumps)
            else:
                x_fine = self.add_jumps(t_fine, old_x_fine, x_fine, current_jumps)
                x_coarse = self.add_jumps(t_coarse, old_x_coarse, x_coarse, current_jumps)

            # store in path
            paths_fine[:, total_steps_coarse] = x_fine
            paths_coarse[:, total_steps_coarse] = x_coarse

            # increment jump index if a jump has just happened
            jump_idxs = torch.where(torch.isclose(next_jump_time, t_fine, atol=1e-12), jump_idxs + 1, jump_idxs)
        return (paths_fine[:, :total_steps_coarse + 1], paths_coarse[:, :total_steps_coarse + 1]), _


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
