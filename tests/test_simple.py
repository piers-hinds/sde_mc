from sde_mc import *
import pytest
import torch.optim


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
def merton_1d_adapted_solver(merton_1d):
    return JumpAdaptedSolver(merton_1d, 3, 10)


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
def sample_jumps_dataloader(merton_1d_solver, euro_call, constant_short_rate):
    return simulate_data(10, merton_1d_solver, euro_call, constant_short_rate, bs=2)


@pytest.fixture
def sample_adapted_dataloader(merton_1d_adapted_solver, euro_call, constant_short_rate):
    return simulate_adapted_data(10, merton_1d_adapted_solver, euro_call, constant_short_rate, bs=2)


@pytest.fixture
def mlps_1d():
    bcv = Mlp(2, [5, 5], 1, activation=nn.ReLU)
    jcv = Mlp(2, [5, 5], 1, activation=nn.ReLU)
    return [bcv, jcv]


@pytest.fixture
def uniform_grid():
    return UniformGrid(0., 3., 10)
