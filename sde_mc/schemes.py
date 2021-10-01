import torch
from .helpers import solve_quadratic


class EulerScheme:
    def step(self, t, x, h, corr_normals):
        if self.sde.diffusion_struct == 'diag':
            new_pos = x + self.sde.drift(t, x) * h + self.sde.diffusion(t, x) * corr_normals
        elif self.sde.diffusion_struct == 'indep':
            new_pos = x + self.sde.drift(t, x) * h + (self.sde.diffusion(t, x) * corr_normals).sum(dim=-1)
        else:
            new_pos = x + self.sde.drift(t, x) * h + torch.matmul(self.sde.diffusion(t, x) * corr_normals)
        return new_pos


class HestonScheme:
    def step(self, t, x, h, corr_normals):
        new_pos = x + self.sde.drift(t, x) * h + self.sde.diffusion(t, x) * corr_normals
        coefs = self.sde.quadratic_parameters(x[:, 1], h, corr_normals[:, 1])
        sol = solve_quadratic(coefs)
        new_pos[:, 1] = sol * sol
        return new_pos