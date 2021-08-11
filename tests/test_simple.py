from sde_mc import *


def test_gbm():
    x0 = torch.tensor([1., 1., 1.])
    gbm = Gbm(0.2, 0.4, x0, dim=3)
    assert gbm.jump_mean() == 0
    assert torch.allclose(gbm.drift(0, x0), 0.2 * x0)
    assert torch.allclose(gbm.diffusion(0, x0), 0.4 * x0)
    assert torch.allclose(gbm.corr_matrix, torch.eye(3))


def test_merton():
    x0 = torch.tensor([1., 1.])
    jumps = torch.tensor([2., 1.])
    size = (3, 4, 2)
    merton = Merton(0.2, 0.4, 2, 0.05, 0.4, x0, dim=2)
    assert torch.allclose(merton.jumps(0, x0, jumps), x0 * jumps)
    assert merton.sample_jumps(size, 'cpu').shape == size

