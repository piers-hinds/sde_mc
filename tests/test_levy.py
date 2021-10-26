from .test_simple import *


def test_inverse_cdf(icdf):
    icdf(torch.tensor(0.5))


def test_exp_example_levy(exp_example_levy):
    t = torch.tensor([1.])
    x = torch.tensor([[1.], [1.]])
    jumps = torch.tensor([[0.], [0.5]])
    assert torch.allclose(exp_example_levy.drift(t, x), exp_example_levy.r * x)
    assert torch.allclose(exp_example_levy.diffusion(t, x), exp_example_levy.sigma * x)
    assert torch.allclose(exp_example_levy.jumps(t, x, jumps), exp_example_levy.f * x * jumps)
