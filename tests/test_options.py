from .test_simple import *


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


def test_digital(terminals_1d):
    digital = Digital(1.)
    assert torch.allclose(digital(terminals_1d), torch.tensor([0., 1., 0., 0.]))


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

