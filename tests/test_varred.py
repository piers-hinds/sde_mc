from .test_simple import *


# varred.py
def test_train_cvs(mlps_1d, sample_jumps_dataloader):
    trials, steps, dim = sample_jumps_dataloader.dataset.paths.shape
    time_points = partition(3, steps, ends='left')
    ys = time_points
    adam = torch.optim.Adam(list(mlps_1d[0].parameters()) + list(mlps_1d[1].parameters()))
    time_elapsed, loss_arr = train_control_variates(mlps_1d, adam, sample_jumps_dataloader, 0.1, 2, time_points, ys,
                                                    epochs=10, print_losses=False)
    assert time_elapsed >= 0
    assert len(loss_arr) == 10


def test_apply_cvs(mlps_1d, sample_jumps_dataloader):
    trials, steps, dim = sample_jumps_dataloader.dataset.paths.shape
    time_points = partition(3, steps, ends='left')
    ys = time_points
    run_sum, run_sum_sq = apply_control_variates(mlps_1d, sample_jumps_dataloader, 0.1, 2, time_points, ys)
    assert run_sum_sq >= 0


def test_train_dcv(mlps_1d, sample_dataloader):
    trials, steps, dim = sample_dataloader.dataset.paths.shape
    time_points = partition(3, steps, ends='left')
    ys = time_points
    adam = torch.optim.Adam(mlps_1d[0].parameters())
    time_elapsed, loss_arr = train_diffusion_control_variate(mlps_1d[0], adam, sample_dataloader, time_points, ys,
                                                             epochs=10, print_losses=False)
    assert time_elapsed >= 0
    assert len(loss_arr) == 10


def test_apply_dcv(mlps_1d, sample_dataloader):
    trials, steps, dim = sample_dataloader.dataset.paths.shape
    time_points = partition(3, steps, ends='left')
    ys = time_points
    run_sum, run_sum_sq = apply_diffusion_control_variate(mlps_1d[0], sample_dataloader, time_points, ys)
    assert run_sum_sq >= 0