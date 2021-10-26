from .test_simple import *


# helpers.py
def test_partition():
    expected = torch.tensor([0., 1., 2., 3.])
    assert torch.allclose(partition(3, 3, 'both'), expected)
    assert torch.allclose(partition(3, 3, 'left'), expected[:-1])
    assert torch.allclose(partition(3, 3, 'right'), expected[1:])
    assert torch.allclose(partition(3, 3, 'none'), expected[1:-1])


def test_quadratic_solver():
    expected = torch.tensor([1., 5., 7., 0.15475321515005])
    a = torch.tensor([1., 1., 1., 2.])
    b = torch.tensor([0., -2., -13., 32.])
    c = torch.tensor([-1., -15., 42., -5.])
    assert torch.allclose(solve_quadratic((a, b, c)), expected)


def test_mc_estimates():
    sample = torch.tensor([1, 2, 3, 4, 5, 20, 100.])
    run_sum, run_sum_sq, n = sample.sum(), (sample*sample).sum(), len(sample)
    expected = (sample.mean(), sample.var())
    result = mc_estimates(run_sum, run_sum_sq, n)
    assert torch.isclose(result[0], expected[0])
    assert torch.isclose(result[1], expected[1])


def test_remove_steps():
    result = remove_steps(0.1, 1000, 3)
    assert result == 966
