from .test_simple import *


# sde.py
def test_gbm(gbm_2d):
    x0 = gbm_2d.init_value
    assert gbm_2d.diffusion_struct == 'diag'
    assert gbm_2d.brown_dim == 2
    assert gbm_2d.jump_rate() == 0
    assert torch.allclose(gbm_2d.drift(0, x0), 0.02 * x0)
    assert torch.allclose(gbm_2d.diffusion(0, x0), 0.2 * x0)
    assert torch.allclose(gbm_2d.corr_matrix, torch.eye(2))


def test_merton(merton_2d):
    x0 = merton_2d.init_value
    jumps = torch.tensor([2., 1.])
    size = (3, 4, 2)
    assert merton_2d.diffusion_struct == 'diag'
    assert merton_2d.brown_dim == 2
    assert torch.allclose(merton_2d.drift(0, x0), (0.02 - merton_2d.jump_mean() * 2) * x0)
    assert torch.allclose(merton_2d.diffusion(0, x0), 0.3 * x0)
    assert torch.allclose(merton_2d.corr_matrix, torch.eye(2))
    assert torch.allclose(merton_2d.jumps(0, x0, jumps), x0 * jumps)
    assert merton_2d.sample_jumps(size, 'cpu').shape == size


def test_heston(heston_1d):
    x = torch.tensor([[1.5, 1], [0.8, 0.9]])
    assert heston_1d.diffusion_struct == 'diag'
    assert heston_1d.brown_dim == 2
    assert heston_1d.jump_rate() == 0
    assert torch.allclose(heston_1d.drift(0, x), torch.tensor([[1.5*0.02, 0], [0.8*0.02, 0]]))
    assert torch.allclose(heston_1d.diffusion(0, x), torch.tensor([[1.5, 0], [0.8 * torch.sqrt(x[1, 1]), 0]]))
    assert torch.allclose(heston_1d.corr_matrix, torch.tensor([[1, -0.5], [-0.5, 1]]))


def test_uniform_grid(uniform_grid):
    grid_partition = torch.tensor([x for x in uniform_grid])
    assert torch.allclose(grid_partition, partition(3, 10, ends='left'))