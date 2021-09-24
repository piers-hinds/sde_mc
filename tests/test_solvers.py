from .test_simple import *


# solvers.py
def test_solvers_gbm(gbm_2d_solver):
    paths, (normals, _) = gbm_2d_solver.solve(bs=4, return_normals=True)
    assert not torch.isnan(paths).any()
    assert paths.shape == (4, 11, 2)
    assert normals.shape == (4, 10, 2)


def test_solvers_merton(merton_1d_solver):
    paths, (normals, jumps) = merton_1d_solver.solve(bs=4, return_normals=True)
    assert not torch.isnan(paths).any()
    assert paths.shape == (4, 11, 1)
    assert normals.shape == (4, 10, 1)
    assert jumps.shape == (4, 10, 1)


def test_solvers_heston(heston_1d_solver):
    paths, (normals, _) = heston_1d_solver.solve(bs=8, return_normals=True)
    assert not torch.isnan(paths).any()
    assert paths.shape == (8, 11, 2)
    assert normals.shape == (8, 10, 2)


def test_multilevel_euler(gbm_1d_solver):
    levels = [4, 8]
    (paths_fine, paths_coarse), norms = gbm_1d_solver.multilevel_euler(bs=4, levels=levels, return_normals=True)


def test_adapted_solver(merton_1d_adapted_solver):
    paths, (normals, jumps, jump_times, time_paths, left_paths, total_steps, jump_paths) = merton_1d_adapted_solver.solve(8)
    assert not torch.isnan(paths).any()
    assert not torch.isnan(normals).any()
    assert not torch.isnan(jumps).any()
    assert not torch.isnan(jump_times).any()
    assert not torch.isnan(time_paths).any()
    assert not torch.isnan(left_paths).any()
    assert not torch.isnan(jump_paths).any()

