import torch
import numpy as np
import time
from .helpers import partition, remove_steps


class EarlyStopping:
    def __init__(self, eps, quantile, cost_batch, alpha):
        self.eps = eps
        self.quantile = quantile
        self.cost_batch = cost_batch
        self.alpha = alpha
        self.cost_epoch = None
        self.batch_size = None

    def threshold(self):
        return self.alpha * (self.cost_epoch * self.eps ** 2 * self.batch_size) / (self.cost_batch * self.quantile ** 2)

    def stop(self, delta_gamma):
        return delta_gamma < self.threshold()


def train_diffusion_control_variate(model, opt, dl, solver, discounter, epochs, print_losses=True, tol=0,
                                    early_stopping=None):
    trials, steps, dim = dl.dataset.paths.shape
    time_points = partition(solver.time_interval, solver.num_steps, ends='left', device=solver.device)
    discounts = discounter(time_points).view(1, len(time_points), 1)
    loss_arr = []

    epoch_total_cost = 0
    start_train = time.time()
    for epoch in range(epochs):
        start_epoch = time.time()
        model.train()
        run_loss = 0
        for (paths, normals), payoffs in dl:
            opt.zero_grad()

            if model.sequential:
                rep_time_points = time_points.unsqueeze(-1).repeat(dl.batch_size, 1, 1)
                f_in = torch.cat([rep_time_points, paths], dim=-1)
            else:
                rep_time_points = time_points.repeat(dl.batch_size).unsqueeze(-1)
                f_in = torch.cat([rep_time_points, paths.reshape(dl.batch_size * steps, dim)], dim=-1)

            f_out = model(f_in).view(dl.batch_size, steps, dim)
            brownians_cv = integrate_cv(normals, f_out, discounts, solver.sde.diffusion_struct, tol=tol,
                                        time_interval=solver.time_interval)
            var_loss = (payoffs + brownians_cv).var()
            run_loss += var_loss.item()
            var_loss.backward()
            opt.step()
        loss_arr.append(run_loss / len(dl))
        if print_losses:
            print('{}: Train loss: {:.5f}     95% confidence interval: {:.5f}'.format(epoch, loss_arr[epoch], np.sqrt(
                loss_arr[epoch]) * 2 / np.sqrt(trials)))
        model.eval()
        end_epoch = time.time()
        epoch_total_cost += end_epoch - start_epoch
        # DECIDE HERE for early stopping - need external information eps, alpha, cost of one batch, var Gamma, cost to sample batch
        if early_stopping is not None and epoch > 0:
            # Set values here
            delta_gamma = loss_arr[epoch - 1] - loss_arr[epoch]
            early_stopping.cost_epoch = epoch_total_cost / (epoch + 1)

            print('dg: {:.5f}    thresh: {:.5f}     time: {:.5f}'.format(delta_gamma, early_stopping.threshold(),
                                                                         early_stopping.cost_epoch))
            if early_stopping.stop(delta_gamma):
                break

    end_train = time.time()
    return end_train - start_train, loss_arr


def apply_diffusion_control_variate(model, dl, solver, discounter, tol=0):
    trials, steps, dim = dl.dataset.paths.shape
    time_points = partition(solver.time_interval, solver.num_steps, ends='left', device=solver.device)
    discounts = discounter(time_points).view(1, len(time_points), 1)

    run_sum, run_sum_sq = 0, 0
    with torch.inference_mode():
        for (paths, normals), payoffs in dl:
            if model.sequential:
                rep_time_points = time_points.unsqueeze(-1).repeat(dl.batch_size, 1, 1)
                f_in = torch.cat([rep_time_points, paths], dim=-1)
            else:
                rep_time_points = time_points.repeat(dl.batch_size).unsqueeze(-1)
                f_in = torch.cat([rep_time_points, paths.reshape(dl.batch_size * steps, dim)], dim=-1)
            f_out = model(f_in).view(dl.batch_size, steps, dim)
            brownians_cv = integrate_cv(normals, f_out, discounts, solver.sde.diffusion_struct, tol=tol,
                                        time_interval=solver.time_interval)
            gammas = payoffs + brownians_cv
            run_sum += gammas.sum()
            run_sum_sq += (gammas * gammas).sum()
    return run_sum, run_sum_sq


def apply_adapted_control_variates(models, dl, solver, discounter, tol=0):
    n, steps, dim = dl.dataset.paths.shape
    f, g = models
    run_sum, run_sum_sq = 0, 0
    with torch.inference_mode():
        for (paths, normals, left_paths, time_paths, jump_paths), payoffs in dl:
            h = torch.diff(time_paths, dim=1)
            discounts = discounter(time_paths)
            if f.sequential:
                f_inputs = torch.cat([time_paths, paths], dim=-1)
            else:
                time_inputs = time_paths.reshape(dl.batch_size * steps, 1)
                paths_inputs = paths.reshape(dl.batch_size * steps, dim)
                f_inputs = torch.cat([time_inputs, paths_inputs], dim=-1)

            f_outputs = f(f_inputs).view(normals.shape)

            brownian_cv = integrate_cv(normals, f_outputs, discounts, solver.sde.diffusion_struct, tol=tol,
                                       time_interval=solver.time_interval)

            if g.sequential:
                g_inputs = torch.cat([time_paths, left_paths], dim=-1)
            else:
                time_inputs = time_paths.reshape(dl.batch_size * steps, 1)
                g_inputs = torch.cat([time_inputs, left_paths.view(dl.batch_size * steps, dim)], dim=-1)
            g_outputs = g(g_inputs).view(dl.batch_size, steps, dim)
            jump_cv = (g_outputs * discounts * jump_paths).sum(-1).sum(-1)

            comps = (- solver.sde.jump_rate() * solver.sde.jump_mean() * g_outputs[:, :-1] *
                     discounts[:, :-1] * h).sum(-1).sum(-1)
            gammas = payoffs + brownian_cv + jump_cv + comps
            run_sum += gammas.sum()
            run_sum_sq += (gammas * gammas).sum()
    return run_sum, run_sum_sq


def train_adapted_control_variates(models, opt, dl, solver, discounter, epochs=10, print_losses=True, tol=0,
                                   early_stopping=None):
    trials, steps, dim = dl.dataset.paths.shape
    loss_arr = []
    f, g = models
    epoch_total_cost = 0

    for epoch in range(epochs):
        start_epoch = time.time()
        f.train()
        g.train()
        run_loss = 0
        for (paths, normals, left_paths, time_paths, jump_paths), payoffs in dl:
            opt.zero_grad()

            h = torch.diff(time_paths, dim=1)
            discounts = discounter(time_paths)
            if f.sequential:
                f_inputs = torch.cat([time_paths, paths], dim=-1)
            else:
                time_inputs = time_paths.reshape(dl.batch_size * steps, 1)
                paths_inputs = paths.reshape(dl.batch_size * steps, dim)
                f_inputs = torch.cat([time_inputs, paths_inputs], dim=-1)

            f_outputs = f(f_inputs).view(normals.shape)

            brownian_cv = integrate_cv(normals, f_outputs, discounts, solver.sde.diffusion_struct, tol=tol,
                                       time_interval=solver.time_interval)

            if g.sequential:
                g_inputs = torch.cat([time_paths, left_paths], dim=-1)
            else:
                time_inputs = time_paths.reshape(dl.batch_size * steps, 1)
                paths_inputs = paths.reshape(dl.batch_size * steps, dim)
                g_inputs = torch.cat([time_inputs, left_paths.view(dl.batch_size * steps, dim)], dim=-1)

            g_outputs = g(g_inputs).view(dl.batch_size, steps, dim)
            jump_cv = (g_outputs * discounts * jump_paths).sum(-1).sum(-1)

            comps = (- solver.sde.jump_rate() * solver.sde.jump_mean() * g_outputs[:, :-1] *
                     discounts[:, :-1] * h).sum(-1).sum(-1)
            gammas = payoffs + brownian_cv + jump_cv + comps

            var_loss = gammas.var()
            run_loss += var_loss.item()
            var_loss.backward()
            opt.step()
        loss_arr.append(run_loss / len(dl))
        if print_losses:
            print('{}: Train loss: {:.5f}     95% confidence interval: {:.5f}'.format(epoch, loss_arr[epoch], np.sqrt(
                loss_arr[epoch]) * 2 / np.sqrt(trials)))
        f.eval()
        g.eval()
        end_epoch = time.time()
        epoch_total_cost += end_epoch - start_epoch
        # DECIDE HERE for early stopping - need external information eps, alpha, cost of one batch, var Gamma, cost to sample batch
        if early_stopping is not None and epoch > 0:
            # Set values here
            delta_gamma = loss_arr[epoch - 1] - loss_arr[epoch]
            early_stopping.cost_epoch = epoch_total_cost / (epoch + 1)

            print('dg: {:.5f}    thresh: {:.5f}     time: {:.5f}'.format(delta_gamma, early_stopping.threshold(),
                                                                         early_stopping.cost_epoch))
            if early_stopping.stop(delta_gamma):
                break
    return loss_arr


def integrate_cv(normals, f_out, discounts, diffusion_struct, tol=0, time_interval=None):
    if tol != 0:
        assert time_interval is not None
        steps = normals.shape[1]
        new_steps = remove_steps(tol, steps, time_interval)
        normals = normals[:, :new_steps]
        f_out = f_out[:, :new_steps]
        discounts = discounts[:, :new_steps]

    if diffusion_struct == 'diag':
        return (normals * f_out * discounts).sum(-1).sum(-1)
    else:
        return ((normals * f_out).sum(-1) * discounts).sum(-1).sum(-1)
