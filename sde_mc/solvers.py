import torch


class SdeSolver:
    def __init__(self, sde, time, num_steps, device='cpu', seed=1):
        """
        :param sde: Sde, the SDE to solve
        :param time: float, the time to solve up to
        :param num_steps: int, the number of steps in the discretization
        :param device: string, the device to do the computations on
        :param seed: int, seed for torch
        """
        self.sde = sde
        self.time = time
        self.num_steps = num_steps
        self.device = device
        self.has_jumps = bool(self.sde.jump_rate())

        if len(self.sde.corr_matrix) > 1:
            self.lower_cholesky = torch.linalg.cholesky(self.sde.corr_matrix.to(device))
        else:
            self.lower_cholesky = torch.tensor([[1.]], device=device)
        torch.manual_seed(seed)

    def sample_corr_normals(self, size, h):
        normals = torch.randn(size=size, device=self.device) * torch.sqrt(h)
        return torch.matmul(self.lower_cholesky, normals)

    def sample_poissons(self, size, h):
        rates = torch.ones(size=size, device=self.device) * (self.sde.rate * h)
        return torch.poisson(rates)

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

        h = torch.tensor(self.time / self.num_steps, device=self.device)

        paths = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)

        paths[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

        corr_normals = self.sample_corr_normals(size=(bs, self.num_steps, self.sde.dim, 1), h=h)

        t = torch.tensor(0.0, device=self.device)

        if self.has_jumps:
            paths_no_jumps = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)
            paths_no_jumps[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

            poissons = self.sample_poissons(size=(bs, self.num_steps, self.sde.dim), h=h)
            max_jumps = self.sample_jumps(size=(bs, self.num_steps, self.sde.dim))
            jumps = (max_jumps * torch.gt(poissons, 0))

            for i in range(self.num_steps):
                paths_no_jumps[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * h + \
                                           self.sde.diffusion(t, paths[:, i]) * corr_normals[:, i].squeeze(-1)
                paths[:, i + 1] = paths_no_jumps[:, i + 1] * (jumps[:, i] + 1)
                t += h
            additional_data = (paths_no_jumps, corr_normals, jumps)
        else:
            for i in range(self.num_steps):
                paths[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * h + \
                                  self.sde.diffusion(t, paths[:, i]) * corr_normals[:, i].squeeze(-1)
                t += h
            additional_data = (None, corr_normals, None)

        if return_normals:
            return paths, additional_data
        else:
            return paths, None


