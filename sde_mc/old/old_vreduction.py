class SdeControlVariate(Sde):
    """An Sde class which adds a control variate to an existing SDE"""

    def __init__(self, base_sde, control_variate, time_points, discounter):
        """
        :param base_sde: Sde, the original SDE to add a control variate to
        :param control_variate: SdeApproximator, the approximation of F * Y
        :param time_points: torch.tensor, the time points at which to update the control variate
        :param discounter: function(float), the discount process Y
        """
        super(SdeControlVariate, self).__init__(torch.cat([base_sde.init_value, torch.tensor([0.])]), base_sde.dim + 1,
                                                base_sde.noise_dim, base_sde.corr_matrix)
        self.base_sde = base_sde
        self.base_dim = base_sde.dim
        self.cv = control_variate
        self.time_points = time_points
        self.discounter = discounter

        self.idx = 0
        self.F = 0

    def drift(self, t, x):
        return torch.cat(
            [self.base_sde.drift(t, x[:, :self.base_dim]), torch.zeros_like(x[:, self.base_dim]).unsqueeze(1)], dim=-1)

    def diffusion(self, t, x):
        if not t:
            self.reset_control(x)
        if t >= self.time_points[self.idx]:
            self.idx += 1
            self.update_control(t, x)

        return torch.cat([self.base_sde.diffusion(t, x[:, :self.base_dim]), self.F], dim=1)

    def update_control(self, t, x):
        """Updates the control variate"""
        # self.F = (self.cv(self.idx, t, x[:, :self.base_dim]) * self.discounter(t)).unsqueeze(-1) *
        # -self.base_sde.diffusion(t, x[:, :self.base_dim])
        self.F = torch.matmul(-torch.transpose(self.base_sde.diffusion(t, x[:, :self.base_dim]), 1, 2),
                              self.cv(self.idx, t, x[:, :self.base_dim]).unsqueeze(-1)).transpose(1, 2) * \
                 self.discounter(t)

    def reset_control(self, x):
        """Resets the control variate (e.g. when restarting)"""
        self.F = torch.zeros_like(x[:, :self.base_dim]).unsqueeze(1)
        self.idx = 0


def train_control_variate(F_approx, dl, opt, time_points, dim, Ys, epochs):
    rep_time_points = time_points.repeat(dl.batch_size).unsqueeze(-1)
    steps = len(time_points)
    loss_arr = []
    for epoch in range(epochs):
        F_approx.train()
        run_loss = 0
        for i, (xb, yb) in enumerate(dl):
            opt.zero_grad()
            inputs = torch.cat([rep_time_points, xb[0].reshape(dl.batch_size*steps, dim)], dim=-1)
            outputs = F_approx(inputs).view(dl.batch_size, steps, dim)
            Zs = (xb[1] * (outputs * Ys.view(1, len(Ys), 1))).sum(-1).sum(-1)
            var_loss = (yb + Zs).var()
            run_loss += var_loss.item()
            var_loss.backward()
            opt.step()
        loss_arr.append(run_loss / len(dl))
        print('{}: Train loss: {:.5f}     Train 95: {:.5f}'.format(epoch, loss_arr[epoch], np.sqrt(loss_arr[epoch])*2 / np.sqrt((i+1)*len(dl))))
        F_approx.eval()


def get_preds(F_approx, new_dl, new_time_points, dim, new_Ys):
    rep_new_time_points = new_time_points.repeat(new_dl.batch_size).unsqueeze(-1)
    new_steps = len(new_time_points)
    run_sum = 0
    run_sum_sq = 0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(new_dl):
            inputs = torch.cat([rep_new_time_points, xb[0].reshape(new_dl.batch_size*new_steps, dim)], dim=-1)
            outputs = F_approx(inputs).view(new_dl.batch_size, new_steps, dim)
            Zs = ((outputs * new_Ys.view(1, len(new_Ys), 1)) * xb[1]).sum(-1).sum(-1)
            gammas = yb + Zs
            run_sum += gammas.sum()
            run_sum_sq += (gammas * gammas).sum()
    new_trials = (i + 1) * new_dl.batch_size
    new_mn = (run_sum / new_trials)
    new_sample_sd = torch.sqrt(((run_sum_sq - (run_sum*run_sum) / new_trials) / (new_trials - 1)))
    new_sd = new_sample_sd / np.sqrt(new_trials)
    print(new_trials)
    print(new_mn, new_sd)
    return new_mn, new_sd