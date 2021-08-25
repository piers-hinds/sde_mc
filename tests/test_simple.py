from sde_mc import *
import pytest


# Useful fixtures
@pytest.fixture
def gbm_1d():
    return Gbm(0.02, 0.2, torch.tensor([1.]), dim=1)


@pytest.fixture
def gbm_2d():
    return Gbm(0.02, 0.2, torch.tensor([1., 2.]), dim=2)


@pytest.fixture
def heston_1d():
    return Heston(0.02, 0.5, 0.1, 0.15, -0.5, torch.tensor([1., 2.]))


@pytest.fixture
def merton_1d():
    return Merton(0.02, 0.3, 2, -0.05, 0.3, torch.tensor([1.]), dim=1)


@pytest.fixture
def merton_2d():
    return Merton(0.02, 0.3, 2, -0.05, 0.3, torch.tensor([1., 1.]), dim=2)


@pytest.fixture
def gbm_1d_solver(gbm_1d):
    return SdeSolver(gbm_1d, 3, 10)


@pytest.fixture
def gbm_2d_solver(gbm_2d):
    return SdeSolver(gbm_2d, 3, 10)


@pytest.fixture
def heston_1d_solver(heston_1d):
    return SdeSolver(heston_1d, 3, 10)


@pytest.fixture
def merton_1d_solver(merton_1d):
    return SdeSolver(merton_1d, 3, 10)


@pytest.fixture
def merton_2d_solver(merton_2d):
    return SdeSolver(merton_2d, 3, 10)


@pytest.fixture
def terminals_1d():
    return torch.tensor([[1.], [3.], [0.5], [0.]])


@pytest.fixture
def terminals_2d():
    return torch.tensor([[1., 2.], [0., 0.5], [1.2, 0.9]])


@pytest.fixture
def euro_call():
    return EuroCall(strike=1)


@pytest.fixture
def constant_short_rate():
    return ConstantShortRate(r=0.02)


@pytest.fixture
def sample_dataloader(gbm_1d_solver, euro_call, constant_short_rate):
    return simulate_data(10, gbm_1d_solver, euro_call, constant_short_rate, bs=2)


@pytest.fixture
def mlps_1d():
    bcv = Mlp(2, [5, 5], 1, activation=nn.ReLU)
    jcv = Mlp(2, [5, 5], 1, activation=nn.ReLU)
    return [bcv, jcv]


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


# sde.py
def test_gbm(gbm_2d):
    x0 = gbm_2d.init_value
    assert gbm_2d.jump_rate() == 0
    assert torch.allclose(gbm_2d.drift(0, x0), 0.02 * x0)
    assert torch.allclose(gbm_2d.diffusion(0, x0), 0.2 * x0)
    assert torch.allclose(gbm_2d.corr_matrix, torch.eye(2))


def test_merton(merton_2d):
    x0 = merton_2d.init_value
    jumps = torch.tensor([2., 1.])
    size = (3, 4, 2)
    assert torch.allclose(merton_2d.drift(0, x0), (0.02 - merton_2d.jump_mean() * 2) * x0)
    assert torch.allclose(merton_2d.diffusion(0, x0), 0.3 * x0)
    assert torch.allclose(merton_2d.corr_matrix, torch.eye(2))
    assert torch.allclose(merton_2d.jumps(0, x0, jumps), x0 * jumps)
    assert merton_2d.sample_jumps(size, 'cpu').shape == size


def test_heston(heston_1d):
    x = torch.tensor([[1.5, 1], [0.8, 0.9]])
    assert heston_1d.jump_rate() == 0
    assert torch.allclose(heston_1d.drift(0, x), torch.tensor([[1.5*0.02, 0], [0.8*0.02, 0]]))
    assert torch.allclose(heston_1d.diffusion(0, x), torch.tensor([[1.5, 0], [0.8 * torch.sqrt(x[1, 1]), 0]]))
    assert torch.allclose(heston_1d.corr_matrix, torch.tensor([[1, -0.5], [-0.5, 1]]))


# options.py
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


def test_euro_call(terminals_1d):
    euro_call = EuroCall(strike=1)
    assert torch.allclose(euro_call(terminals_1d), torch.tensor([0., 2., 0., 0.]))


def test_binary_aon(terminals_1d):
    binary_aon = BinaryAoN(strike=1)
    assert torch.allclose(binary_aon(terminals_1d), torch.tensor([1., 3., 0., 0.]))


def test_basket(terminals_2d):
    basket = Basket(strike=1)
    assert torch.allclose(basket(terminals_2d), torch.tensor([0.5, 0, 0.05]))


def test_rainbow(terminals_2d):
    rainbow = Rainbow(strike=1)
    assert torch.allclose(rainbow(terminals_2d), torch.tensor([1., 0., 0.2]))


def test_constant_short_rate():
    time_points = torch.tensor([0., 1., 2.])
    constant_short_rate = ConstantShortRate(r=0.02)
    discounts = constant_short_rate(time_points)
    assert torch.allclose(discounts, (time_points * -0.02).exp())


# solvers.py
def test_solvers_gbm(gbm_2d_solver):
    paths, (_, normals, _) = gbm_2d_solver.solve(bs=4, return_normals=True)
    assert not torch.isnan(paths).any()
    assert paths.shape == (4, 11, 2)
    assert normals.shape == (4, 10, 2)


def test_solvers_merton(merton_1d_solver):
    paths, (left_paths, normals, jumps) = merton_1d_solver.solve(bs=4, return_normals=True)
    assert not torch.isnan(paths).any()
    assert paths.shape == (4, 11, 1)
    assert left_paths.shape == (4, 11, 1)
    assert normals.shape == (4, 10, 1)
    assert jumps.shape == (4, 10, 1)
    all_jumps = torch.cat([torch.zeros_like(paths[:, :1, :]), jumps], dim=1)
    #assert torch.allclose(all_jumps * left_paths + left_paths, paths)


def test_multilevel_euler(gbm_1d_solver):
    levels = [4, 8]
    (paths_fine, paths_coarse), norms = gbm_1d_solver.multilevel_euler(bs=4, levels=levels, return_normals=True)


# varred.py
def test_apply_dcv(mlps_1d, sample_dataloader):
    trials, steps, dim = sample_dataloader.dataset.paths.shape
    time_points = partition(3, steps, ends='left')
    ys = time_points
    run_sum, run_sum_sq = apply_diffusion_control_variate(mlps_1d[0], sample_dataloader, time_points, ys)
    assert run_sum_sq >= 0


# mc.py
def test_mc_simple(gbm_2d_solver):
    mc_stats = mc_simple(100, gbm_2d_solver, EuroCall(1), ConstantShortRate(0.02))
    assert mc_stats.time_elapsed >= 0
    assert mc_stats.sample_mean >= 0
    assert mc_stats.sample_std > 0
    assert not mc_stats.payoffs.isnan().any()
    assert mc_stats.paths.shape == (100, 11, 2)


def test_mc_multilevel(gbm_1d_solver):
    mc_stats = mc_multilevel([50, 20, 10], [1, 4, 16], gbm_1d_solver, EuroCall(1), ConstantShortRate(0.02))
    assert mc_stats.time_elapsed >= 0
    assert mc_stats.sample_mean >= 0
    assert mc_stats.sample_std > 0


def test_get_optimal_trials(gbm_1d_solver):
    opt_trials = get_optimal_trials(100, [1, 4, 16], 0.001, gbm_1d_solver, EuroCall(1), ConstantShortRate(0.02))
    assert len(opt_trials) == 3





