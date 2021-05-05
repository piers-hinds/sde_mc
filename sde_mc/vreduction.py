from .sde import Sde
import torch


class SdeControlVariate(Sde):
    """An Sde class which adds a control variate to an existing SDE"""

    def __init__(self, base_sde, control_variate, time_points):
        """
        :param base_sde: Sde, the original SDE to add a control variate to
        :param control_variate: SdeApproximator, the approximation of F * Y
        :param time_points: torch.tensor, the time points at which to update the control variate
        """
        super(SdeControlVariate, self).__init__(torch.cat([base_sde.init_value, torch.tensor([0.])]), base_sde.dim + 1,
                                                base_sde.noise_dim, base_sde.corr_matrix)
        self.base_sde = base_sde
        self.base_dim = base_sde.dim
        self.cv = control_variate
        self.time_points = time_points

        self.idx = 0
        self.F = 0

    def drift(self, t, x):
        return torch.cat([self.base_sde.drift(t, x[:, :self.base_dim]), torch.zeros_like(x[:, self.base_dim]).unsqueeze(1)], dim=-1)

    def diffusion(self, t, x):
        if not t:
            self.reset_control(x)
        if t >= self.time_points[self.idx]:
            self.idx += 1
            self.update_control(t, x)

        return torch.cat([self.base_sde.diffusion(t, x[:, :self.base_dim]), self.F], dim=1)

    def update_control(self, t, x):
        """Updates the control variate"""
        self.F = self.cv(self.idx, t, x[:, :self.base_dim]).unsqueeze(1)

    def reset_control(self, x):
        """Resets the control variate (e.g. when restarting)"""
        self.F = torch.zeros_like(x[:, :self.base_dim]).unsqueeze(1)
        self.idx = 0
