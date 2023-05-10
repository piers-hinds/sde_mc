from .test_simple import *


# mc.py
def test_mc_simple(gbm_2d_solver):
    mc_stats = mc_simple(100, gbm_2d_solver, EuroCall(1), ConstantShortRate(0.02))
    assert mc_stats.time_elapsed >= 0
    assert mc_stats.sample_mean >= 0
    assert mc_stats.sample_std > 0
    assert not mc_stats.payoffs.isnan().any()
    assert mc_stats.paths.shape == (100, 11, 2)


def test_mc_simple_batch(gbm_1d_solver, euro_call, constant_short_rate):
    mc_stats = mc_simple(100, gbm_1d_solver, euro_call, constant_short_rate, bs=17)
    assert mc_stats.time_elapsed >= 0
    assert mc_stats.sample_mean >= 0
    assert mc_stats.sample_std > 0


def test_mc_control_variates(gbm_1d_solver, mlps_1d, euro_call, constant_short_rate):
    adam = torch.optim.Adam(mlps_1d[0].parameters())
    mc_stats = mc_control_variates(mlps_1d[0], adam, gbm_1d_solver, (100, 100), (10, 20), euro_call, constant_short_rate,
                                   sim_bs=(100, 100), bs=(10, 10), print_losses=False)


# def test_mc_multilevel(gbm_1d_solver):
#     mc_stats = mc_multilevel([50, 20, 10], [1, 4, 16], gbm_1d_solver, EuroCall(1), ConstantShortRate(0.02))
#     assert mc_stats.time_elapsed >= 0
#     assert mc_stats.sample_mean >= 0
#     assert mc_stats.sample_std > 0


# def test_get_optimal_trials(gbm_1d_solver):
#     opt_trials = get_optimal_trials(100, [1, 4, 16], 0.001, gbm_1d_solver, EuroCall(1), ConstantShortRate(0.02))
#     assert len(opt_trials) == 3


def test_simulate_data(gbm_2d_solver):
    dl = simulate_data(16, gbm_2d_solver, EuroCall(1), ConstantShortRate(0.02), bs=16)


def test_simulate_data_jumps(merton_1d_solver):
    dl = simulate_data(16, merton_1d_solver, EuroCall(1), ConstantShortRate(0.02), bs=16)


def test_simulate_adapted(merton_1d_solver, euro_call, constant_short_rate):
    dl = simulate_adapted_data(16, merton_1d_solver, euro_call, constant_short_rate, bs=8)


def test_mc_adaptive_cv(merton_1d_solver, mlps_1d, euro_call, constant_short_rate):
    adam = torch.optim.Adam(list(mlps_1d[0].parameters()) + list(mlps_1d[1].parameters()))
    mc_stats = mc_adaptive_cv(mlps_1d, adam, merton_1d_solver, (100, 100), (10, 20), euro_call,
                              constant_short_rate, sim_bs=(100, 100), bs=(10, 10), print_losses=False)


def test_mc_terminal_cv(gbm_1d_solver, euro_call, constant_short_rate):
    mc_stats = mc_terminal_cv(100, gbm_1d_solver, euro_call, constant_short_rate, 10)


def test_run_mc_terminal_cv(heston_problem):
    mc_stats = run_mc_terminal_cv(heston_problem, 0.01, bs=1e3, init_trials=1e4)

