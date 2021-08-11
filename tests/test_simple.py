from sde_mc import *


# helpers
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


# sde
def test_gbm():
    x0 = torch.tensor([1., 1., 1.])
    gbm = Gbm(0.2, 0.4, x0, dim=3)
    assert gbm.jump_rate() == 0
    assert torch.allclose(gbm.drift(0, x0), 0.2 * x0)
    assert torch.allclose(gbm.diffusion(0, x0), 0.4 * x0)
    assert torch.allclose(gbm.corr_matrix, torch.eye(3))


def test_merton():
    x0 = torch.tensor([1., 4.])
    jumps = torch.tensor([2., 1.])
    size = (3, 4, 2)
    merton = Merton(0.2, 0.4, 2, 0.05, 0.4, x0, dim=2)
    assert torch.allclose(merton.drift(0, x0), (0.2 - merton.jump_mean() * 2) * x0)
    assert torch.allclose(merton.diffusion(0, x0), 0.4 * x0)
    assert torch.allclose(merton.corr_matrix, torch.eye(2))
    assert torch.allclose(merton.jumps(0, x0, jumps), x0 * jumps)
    assert merton.sample_jumps(size, 'cpu').shape == size


def test_heston():
    x0 = torch.tensor([1., 2.])
    x = torch.tensor([[1.5, 1], [0.8, 0.9]])
    heston = Heston(0.02, 0.5, 0.1, 0.15, -0.5, x0)
    assert heston.jump_rate() == 0
    assert torch.allclose(heston.drift(0, x), torch.tensor([[1.5*0.02, 0], [0.8*0.02, 0]]))
    assert torch.allclose(heston.diffusion(0, x), torch.tensor([[1.5, 0], [0.8 * torch.sqrt(x[1, 1]), 0]]))
    assert torch.allclose(heston.corr_matrix, torch.tensor([[1, -0.5], [-0.5, 1]]))


# options
def test_bs_binary_aon():
    expected = torch.tensor(0.63548275523).double()
    output = torch.tensor(bs_binary_aon(1, 1, 3, 0.02, 0.2))
    assert torch.isclose(output, expected)


def test_bs_call():
    expected = torch.tensor(0.1646004265).double()
    output = torch.tensor(bs_call(1, 1, 3, 0.02, 0.2))
    assert torch.isclose(output, expected)


def test_merton_call():
    expected = torch.tensor(0.36328189504657027).double()
    output = torch.tensor(merton_call(1, 1, 3, 0.02, 0.3, -0.05, 0.3, 2))
    assert torch.isclose(output, expected)


def test_options():
    terminals = torch.tensor([[1.], [3.], [0.5], [0.]])
    multi_terminals = torch.tensor([[1., 2.], [0., 0.5], [1.2, 0.9]])
    euro_call = EuroCall(strike=1)
    binary_aon = BinaryAoN(strike=1)
    basket = Basket(strike=1)
    rainbow = Rainbow(strike=1)
    assert torch.allclose(euro_call(terminals), torch.tensor([0., 2., 0., 0.]))
    assert torch.allclose(binary_aon(terminals), torch.tensor([1., 3., 0., 0.]))
    assert torch.allclose(basket(multi_terminals), torch.tensor([0.5, 0, 0.05]))
    assert torch.allclose(rainbow(multi_terminals), torch.tensor([1., 0., 0.2]))


def test_discounters():
    time_points = torch.tensor([0., 1., 2.])
    constant_short_rate = ConstantShortRate(r=0.02)
    discounts = constant_short_rate(time_points)
    assert torch.allclose(discounts, (time_points * -0.02).exp())


# solvers
def test_solvers():
    gbm = Gbm(0.02, 0.2, torch.tensor([1., 2.]), dim=2)
    solver = SdeSolver(gbm, 3, 10)
    paths, (_, normals, _) = solver.solve(bs=4, return_normals=True)
    assert not torch.isnan(paths).any()
    assert paths.shape == (4, 11, 2)
    assert normals.shape == (4, 10, 2, 1)

    merton = Merton(0.02, 0.3, 2, -0.05, 0.3, torch.tensor([1.]), dim=1)
    solver = SdeSolver(merton, 3, 10)
    paths,  (left_paths, normals, jumps) = solver.solve(bs=4, return_normals=True)
    assert not torch.isnan(paths).any()
    assert paths.shape == (4, 11, 1)
    assert left_paths.shape == (4, 11, 1)
    assert normals.shape == (4, 10, 1, 1)
    assert jumps.shape == (4, 10, 1)
    all_jumps = torch.cat([torch.zeros_like(paths[:, :1, :]), jumps], dim=1)
    assert torch.allclose(all_jumps * left_paths + left_paths, paths)


# mc
def test_mc_simple():
    gbm = Gbm(0.02, 0.2, torch.tensor([1.]), dim=1)
    solver = SdeSolver(gbm, 3, 10)
    mc_stats = mc_simple(100, solver, EuroCall(1), ConstantShortRate(0.02)(solver.time))
    assert mc_stats.time_elapsed > 0
    assert mc_stats.sample_mean > 0
    assert mc_stats.sample_std > 0
    assert not mc_stats.payoffs.isnan().any()
    assert mc_stats.paths.shape == (100, 11, 1)







