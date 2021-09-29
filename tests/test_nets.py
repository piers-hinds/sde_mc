import torch

from .test_simple import *


# nets.py
def test_zero_function():
    f = ZeroFunction(26)
    x = torch.randn((12, 15))
    assert torch.allclose(f(x), torch.zeros((12, 26)))


def test_lstm():
    x = torch.tensor([[[1., 2.], [3., 4.]]])
    lstm = Lstm(2, 20, 1)
    out = lstm(x)


def test_normal_path_data(gbm_2d_solver):
    mc_stats = mc_simple(16, gbm_2d_solver, EuroCall(1), ConstantShortRate(0.02), return_normals=True)
    data = NormalPathData(mc_stats.paths, mc_stats.payoffs, mc_stats.normals[0])


def test_normal_jumps_path_data(merton_1d_solver):
    mc_stats = mc_simple(16, merton_1d_solver, EuroCall(1), ConstantShortRate(0.02), return_normals=True)
    data = NormalJumpsPathData(mc_stats.paths, mc_stats.payoffs, mc_stats.normals[0], mc_stats.normals[1])
