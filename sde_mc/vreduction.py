import torch
import numpy as np
import time


def apply_control_variates(models, dl, jump_mean, rate, time_points, Ys):
    f, g = models
    n, steps, dim = dl.dataset.paths.shape
    run_sum = 0
    run_sum_sq = 0
    h = time_points[1] - time_points[0]
    discounts = Ys.view(1, len(Ys), 1)
    rep_time_points = time_points.repeat(dl.batch_size).unsqueeze(-1)

    inference_start = time.time()
    with torch.no_grad():
        for xb, yb in dl:
            inputs = torch.cat([rep_time_points, xb[1].reshape(dl.batch_size * steps, dim)], dim=-1)
            g_out = g(inputs).view(dl.batch_size, steps, dim)
            f_out = f(inputs).view(dl.batch_size, steps, dim)

            Zs = (xb[2] * f_out * discounts).sum(-1).sum(-1)
            Js = (xb[3] * g_out * discounts).sum(-1).sum(-1)
            comps = (- rate * jump_mean * g_out * discounts * h).sum(-1).sum(-1)

            gammas = (yb + Js + comps + Zs)
            run_sum += gammas.sum()
            run_sum_sq += (gammas * gammas).sum()
        sample_mean = run_sum / n
        sample_std = torch.sqrt(((run_sum_sq - (run_sum * run_sum) / n) / (n - 1)))

    mc_error = sample_std / np.sqrt(n)
    inference_end = time.time()
    return sample_mean, mc_error, inference_end - inference_start, (run_sum, run_sum_sq)


def train_control_variates(models, opt, dl, jump_mean, rate, time_points, Ys, epochs):
    f, g = models
    trials, steps, dim = dl.dataset.paths.shape
    rep_time_points = time_points.repeat(dl.batch_size).unsqueeze(-1)
    h = time_points[1] - time_points[0]
    loss_arr = []
    dim = dl.dataset.paths.shape[2]
    discounts = Ys.view(1, len(Ys), 1)

    start_train = time.time()
    for epoch in range(epochs):
        f.train(); g.train()
        run_loss = 0
        for i, (xb, yb) in enumerate(dl):
            opt.zero_grad()
            f_in = torch.cat([rep_time_points, xb[0].reshape(dl.batch_size*steps, dim)], dim=-1)
            g_in = torch.cat([rep_time_points, xb[1].reshape(dl.batch_size*steps, dim)], dim=-1)
            f_out = f(f_in).view(dl.batch_size, steps, dim)
            g_out = g(g_in).view(dl.batch_size, steps, dim)

            brownians_cv = (xb[2] * f_out * discounts).sum(-1).sum(-1)
            jumps_cv = (xb[3] * g_out * discounts).sum(-1).sum(-1)
            comps = (- rate * jump_mean * g_out * discounts * h).sum(-1).sum(-1)

            var_loss = (yb + jumps_cv + comps + brownians_cv).var()
            run_loss += var_loss.item()
            var_loss.backward()
            opt.step()
        loss_arr.append(run_loss / len(dl))
        print('{}: Train loss: {:.5f}     95% confidence interval: {:.5f}'.format(epoch, loss_arr[epoch], np.sqrt(loss_arr[epoch])*2 / np.sqrt(trials)))
        f.eval(); g.eval()
    end_train = time.time()
    return end_train-start_train, loss_arr