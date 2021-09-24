import torch
import numpy as np
import time
from .helpers import partition


def train_diffusion_control_variate(model, opt, dl, solver, discounter, epochs, print_losses):
    trials, steps, dim = dl.dataset.paths.shape
    time_points = partition(solver.time_interval, solver.num_steps, ends='left', device=solver.device)

    discounts = discounter(time_points).view(1, len(time_points), 1)
    loss_arr = []

    start_train = time.time()
    for epoch in range(epochs):
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
            brownians_cv = (normals * f_out * discounts).sum(-1).sum(-1)
            var_loss = (payoffs + brownians_cv).var()
            run_loss += var_loss.item()
            var_loss.backward()
            opt.step()
        loss_arr.append(run_loss / len(dl))
        if print_losses:
            print('{}: Train loss: {:.5f}     95% confidence interval: {:.5f}'.format(epoch, loss_arr[epoch], np.sqrt(
                loss_arr[epoch]) * 2 / np.sqrt(trials)))
        model.eval()
    end_train = time.time()
    return end_train - start_train, loss_arr


def apply_diffusion_control_variate(model, dl, solver, discounter):
    trials, steps, dim = dl.dataset.paths.shape
    time_points = partition(solver.time_interval, solver.num_steps, ends='left', device=solver.device)
    rep_time_points = time_points.unsqueeze(-1).repeat(dl.batch_size, 1, 1)
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
            brownians_cv = (normals * f_out * discounts).sum(-1).sum(-1)
            gammas = payoffs + brownians_cv
            run_sum += gammas.sum()
            run_sum_sq += (gammas * gammas).sum()
    return run_sum, run_sum_sq


def train_control_variates(models, opt, dl, solver, discounter, epochs, print_losses=True):
    f, g = models
    trials, steps, dim = dl.dataset.paths.shape
    time_points = partition(solver.time_interval, solver.num_steps, ends='left', device=solver.device)
    h = time_points[1] - time_points[0]
    loss_arr = []
    discounts = discounter(time_points).view(1, len(time_points), 1)

    start_train = time.time()
    for epoch in range(epochs):
        f.train(); g.train()
        run_loss = 0
        for i, ((paths, normals, jumps), payoffs) in enumerate(dl):
            opt.zero_grad()
            if f.sequential:
                rep_time_points = time_points.unsqueeze(-1).repeat(dl.batch_size, 1, 1)
                f_in = torch.cat([rep_time_points, paths], dim=-1)
                g_in = torch.cat([rep_time_points, paths], dim=-1)
            else:
                rep_time_points = time_points.repeat(dl.batch_size).unsqueeze(-1)
                f_in = torch.cat([rep_time_points, paths.reshape(dl.batch_size * steps, dim)], dim=-1)
                g_in = torch.cat([rep_time_points, paths.reshape(dl.batch_size * steps, dim)], dim=-1)

            f_out = f(f_in).view(dl.batch_size, steps, dim)
            g_out = g(g_in).view(dl.batch_size, steps, dim)

            brownians_cv = (normals * f_out * discounts).sum(-1).sum(-1)
            jumps_cv = (jumps * g_out * discounts).sum(-1).sum(-1)
            comps = (- solver.sde.jump_rate() * solver.sde.jump_mean() * g_out * discounts * h).sum(-1).sum(-1)

            var_loss = (payoffs + jumps_cv + comps + brownians_cv).var()
            run_loss += var_loss.item()
            var_loss.backward()
            opt.step()
        loss_arr.append(run_loss / len(dl))
        if print_losses:
            print('{}: Train loss: {:.5f}     95% confidence interval: {:.5f}'.format(epoch, loss_arr[epoch], np.sqrt(loss_arr[epoch])*2 / np.sqrt(trials)))
        f.eval(); g.eval()
    end_train = time.time()
    return end_train-start_train, loss_arr


def apply_control_variates(models, dl, solver, discounter):
    f, g = models
    trials, steps, dim = dl.dataset.paths.shape
    time_points = partition(solver.time_interval, solver.num_steps, ends='left', device=solver.device)
    h = time_points[1] - time_points[0]
    discounts = discounter(time_points).view(1, len(time_points), 1)

    run_sum, run_sum_sq = 0, 0
    with torch.inference_mode():
        for (paths, normals, jumps), payoffs in dl:
            if f.sequential:
                rep_time_points = time_points.unsqueeze(-1).repeat(dl.batch_size, 1, 1)
                f_in = torch.cat([rep_time_points, paths], dim=-1)
                g_in = torch.cat([rep_time_points, paths], dim=-1)
            else:
                rep_time_points = time_points.repeat(dl.batch_size).unsqueeze(-1)
                f_in = torch.cat([rep_time_points, paths.reshape(dl.batch_size * steps, dim)], dim=-1)
                g_in = torch.cat([rep_time_points, paths.reshape(dl.batch_size * steps, dim)], dim=-1)

            f_out = f(f_in).view(dl.batch_size, steps, dim)
            g_out = g(g_in).view(dl.batch_size, steps, dim)

            brownians_cv = (normals * f_out * discounts).sum(-1).sum(-1)
            jumps_cv = (jumps * g_out * discounts).sum(-1).sum(-1)
            comps = (- solver.sde.jump_rate() * solver.sde.jump_mean() * g_out * discounts * h).sum(-1).sum(-1)

            gammas = payoffs + brownians_cv + jumps_cv + comps
            run_sum += gammas.sum()
            run_sum_sq += (gammas * gammas).sum()
    return run_sum, run_sum_sq


def apply_adapted_control_variates(models, dl, solver, discounter):
    n, steps, dim = dl.dataset.paths.shape
    total_steps = dl.dataset.total_steps
    f, g = models
    run_sum, run_sum_sq = 0, 0
    with torch.no_grad():
        for (paths, normals, left_paths, time_paths, jump_paths), payoffs in dl:
            h = torch.diff(time_paths, dim=1)
            discounts = discounter(time_paths)
            time_inputs = time_paths.reshape(dl.batch_size * steps, dim)
            paths_inputs = paths.reshape(dl.batch_size * steps, dim)

            f_inputs = torch.cat([time_inputs, paths_inputs], dim=-1)
            f_outputs = f(f_inputs).view(dl.batch_size, steps, dim)
            brownian_cv = (normals * f_outputs).sum(-1).sum(-1)

            g_inputs = torch.cat([time_inputs, left_paths.view(dl.batch_size * steps, dim)], dim=-1)
            g_outputs = g(g_inputs).view(dl.batch_size, steps, dim)
            jump_cv = (g_outputs * discounts * jump_paths).sum(-1).sum(-1)

            comps = (- solver.sde.jump_rate() * solver.sde.jump_mean() * g_outputs[:, :-1] *
                     discounts[:, :-1] * h).sum(-1).sum(-1)
            gammas = payoffs + brownian_cv + jump_cv + comps
            run_sum += gammas.sum()
            run_sum_sq += (gammas * gammas).sum()
    return run_sum, run_sum_sq


def train_adapted_control_variates(models, opt, dl, solver, discounter, epochs=10, print_losses=True):
    trials, steps, dim = dl.dataset.paths.shape
    loss_arr = []
    f, g = models
    for epoch in range(epochs):
        f.train(), g.train()
        run_loss = 0
        for (paths, normals, left_paths, time_paths, jump_paths), payoffs in dl:
            opt.zero_grad()

            h = torch.diff(time_paths, dim=1)
            discounts = discounter(time_paths)

            time_inputs = time_paths.reshape(dl.batch_size * steps, dim)
            paths_inputs = paths.reshape(dl.batch_size * steps, dim)

            f_inputs = torch.cat([time_inputs, paths_inputs], dim=-1)
            f_outputs = f(f_inputs).view(dl.batch_size, steps, dim)
            brownian_cv = (normals * f_outputs).sum(-1).sum(-1)

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
        f.eval(); g.eval()
    return None
