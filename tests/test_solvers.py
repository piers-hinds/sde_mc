from .test_simple import *


# solvers.py
def test_solvers_gbm(gbm_2d_solver):
    paths, normals = gbm_2d_solver.solve(bs=4, return_normals=True)
    assert not torch.isnan(paths).any()
    assert paths.shape == (4, 11, 2)
    assert normals.shape == (4, 10, 2)


def test_solvers_heston(heston_1d_solver):
    paths, normals = heston_1d_solver.solve(bs=8, return_normals=True)
    assert not torch.isnan(paths).any()
    assert paths.shape == (8, 11, 2)
    assert normals.shape == (8, 10, 2)


def test_adapted_solver(merton_1d_solver):
    paths, (normals, time_paths, left_paths, total_steps, jump_paths) = merton_1d_solver.solve(8)
    assert not torch.isnan(paths).any()
    assert not torch.isnan(normals).any()
    assert not torch.isnan(time_paths).any()
    assert not torch.isnan(left_paths).any()
    assert not torch.isnan(jump_paths).any()


