import torch


class SdeSolver:
    """A class for solving SDEs"""

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

        if len(self.sde.corr_matrix) > 1:
            self.lower_cholesky = torch.linalg.cholesky(self.sde.corr_matrix.to(device))
        else:
            self.lower_cholesky = torch.tensor([[1.]], device=device)
        torch.manual_seed(seed)

    def euler(self, bs=1, return_normals=False, input_normals=None):
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
        
        if input_normals is None:
            normals = torch.randn(size=(bs, self.num_steps, self.sde.noise_dim, 1), device=self.device) * torch.sqrt(h)
            corr_normals = torch.matmul(self.lower_cholesky, normals)
        else:
            corr_normals = input_normals
        
        t = torch.tensor(0.0, device=self.device)
        for i in range(self.num_steps):
            paths[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * h + \
                              torch.matmul(self.sde.diffusion(t, paths[:, i]), corr_normals[:, i]).squeeze(-1)
            t += h

        if return_normals:
            return paths, corr_normals
        else:
            return paths, None


class FastSdeSolver(SdeSolver):
    def euler(self, bs=1, return_normals=False, input_normals=None):
        assert bs >=1
        bs = int(bs)
        h = torch.tensor(self.time / self.num_steps, device=self.device)

        paths = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)
        paths[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        
        if input_normals is None:
            normals = torch.randn(size=(bs, self.num_steps, self.sde.dim, 1), device=self.device) * torch.sqrt(h)
            corr_normals = torch.matmul(self.lower_cholesky, normals)
        else:
            corr_normals = input_normals
        
        t = torch.tensor(0.0, device=self.device)
        for i in range(self.num_steps):
            paths[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * h + self.sde.diffusion(t, paths[:, i]) * corr_normals[:, i].squeeze(-1)

            t += h

        if return_normals:
            return paths, corr_normals
        else:
            return paths, None


class JumpSolver(SdeSolver):
    """A class for solving SDEs which have jumps"""

    def __init__(self, sde, time, num_steps, device='cpu', seed=1):
        """
        :param sde: SdeJumps, the SDE to solve
        :param time: float, the time to solve up to
        :param num_steps: int, the number of steps in the discretization
        :param device: string, the device to do the computations on
        :param seed: int, seed for torch
        """
        super(JumpSolver, self).__init__(sde, time, num_steps, device, seed)
        self.rate = sde.rate
        self.m = sde.mean
        self.v = sde.std

    def euler(self, bs=1, return_normals=False):
        assert bs >= 1, "Batch size must at least one"
        bs = int(bs)

        h = torch.tensor(self.time / self.num_steps, device=self.device)

        paths = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)
        paths_no_jumps = torch.empty(size=(bs, self.num_steps + 1, self.sde.dim), device=self.device)

        paths[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)
        paths_no_jumps[:, 0] = self.sde.init_value.unsqueeze(0).repeat(bs, 1).to(self.device)

        normals = torch.randn(size=(bs, self.num_steps, self.sde.noise_dim, 1), device=self.device) * torch.sqrt(h)
        corr_normals = torch.matmul(self.lower_cholesky, normals)

        rates = torch.ones(size=(bs, self.num_steps, self.sde.dim), device=self.device) * (self.rate * h)
        poissons = torch.poisson(rates)
        max_jumps = (torch.randn(size=(bs, self.num_steps, self.sde.dim), device=self.device) *
                     self.v + self.m).exp() - 1
        jumps = (max_jumps * torch.gt(poissons, 0))

        t = torch.tensor(0.0, device=self.device)
        for i in range(self.num_steps):
            paths_no_jumps[:, i + 1] = paths[:, i] + self.sde.drift(t, paths[:, i]) * h + \
                                       torch.matmul(self.sde.diffusion(t, paths[:, i]), corr_normals[:, i]).squeeze(-1)
            paths[:, i + 1] = paths_no_jumps[:, i + 1] * (jumps[:, i] + 1)
            t += h

        if return_normals:
            return paths, (paths_no_jumps, corr_normals, jumps)
        else:
            return paths, None

