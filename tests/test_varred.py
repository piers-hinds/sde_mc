from .test_simple import *


# varred.py
def test_train_cvs(mlps_1d, sample_jumps_dataloader, merton_1d_solver, constant_short_rate):
    adam = torch.optim.Adam(list(mlps_1d[0].parameters()) + list(mlps_1d[1].parameters()))
    time_elapsed, loss_arr = train_control_variates(mlps_1d, adam, sample_jumps_dataloader, merton_1d_solver,
                                                    constant_short_rate, epochs=10, print_losses=False)
    assert time_elapsed >= 0
    assert len(loss_arr) == 10


def test_apply_cvs(mlps_1d, sample_jumps_dataloader, merton_1d_solver, constant_short_rate):
    run_sum, run_sum_sq = apply_control_variates(mlps_1d, sample_jumps_dataloader, merton_1d_solver, constant_short_rate)
    assert run_sum_sq >= 0


def test_train_dcv(mlps_1d, sample_dataloader, gbm_1d_solver, constant_short_rate):
    adam = torch.optim.Adam(mlps_1d[0].parameters())
    time_elapsed, loss_arr = train_diffusion_control_variate(mlps_1d[0], adam, sample_dataloader, gbm_1d_solver,
                                                             constant_short_rate, epochs=10, print_losses=False)
    assert time_elapsed >= 0
    assert len(loss_arr) == 10


def test_apply_dcv(mlps_1d, sample_dataloader, gbm_1d_solver, constant_short_rate):
    run_sum, run_sum_sq = apply_diffusion_control_variate(mlps_1d[0], sample_dataloader, gbm_1d_solver,
                                                          constant_short_rate)
    assert run_sum_sq >= 0


def test_train_adapted(mlps_1d, sample_adapted_dataloader, merton_1d_adapted_solver, constant_short_rate):
    adam = torch.optim.Adam(list(mlps_1d[0].parameters()) + list(mlps_1d[1].parameters()))
    _ = train_adapted_control_variates(mlps_1d, adam, sample_adapted_dataloader, merton_1d_adapted_solver,
                                       constant_short_rate, print_losses=False)


def test_apply_adapted(mlps_1d, sample_adapted_dataloader, merton_1d_adapted_solver, constant_short_rate):
    apply_adapted_control_variates(mlps_1d, sample_adapted_dataloader, merton_1d_adapted_solver, constant_short_rate)
