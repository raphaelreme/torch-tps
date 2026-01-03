import math

import pytest
import torch

from torch_tps.polynomial_transform import PolynomialFeatures


def generate_random_inputs(n: int, d: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(0)
    return torch.randn((n, d), generator=rng)


def test_degree_1_valid_in_2d():
    inputs = torch.tensor([[2.0, 3.0]])
    pf = PolynomialFeatures(degree=1).fit(inputs)
    transformed = pf.transform(inputs)

    # degree 1 => [1, a, b]
    expected = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(transformed, expected)


def test_degree_2_valid_in_2d():
    inputs = torch.tensor([[2.0, 3.0]])
    pf = PolynomialFeatures(degree=2).fit(inputs)
    predicted = pf.transform(inputs)

    # degree 2 => [1, a, b, a^2, ab, b^2]
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]])
    assert torch.allclose(predicted, expected)


@pytest.mark.parametrize(("n", "d", "degree"), [(5, 4, 3), (20, 2, 2), (1, 1, 1), (2, 1, 4), (3, 3, 1)])
def test_output_dim_matches_combinatorial_count(n: int, d: int, degree: int):
    # d_out should be C(d+degree, degree)
    inputs = generate_random_inputs(n, d)
    pf = PolynomialFeatures(degree).fit(inputs)
    expected = math.comb(d + degree, degree)

    predicted = pf.transform(inputs)
    assert predicted.shape == (n, expected)


def test_deterministic_and_order_stable():
    n = 10
    d = 3
    degree = 2
    inputs_1 = generate_random_inputs(n, d)
    inputs_2 = inputs_1.clone()

    predicted_1 = PolynomialFeatures(degree).fit(inputs_1).transform(inputs_1)
    predicted_2 = PolynomialFeatures(degree).fit(inputs_2).transform(inputs_2)

    assert torch.allclose(predicted_1, predicted_2), "PolynomialFeatures should be deterministic and stable"


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float64])
def test_dtype_conserved(dtype: torch.dtype):
    # NOTE: integer types are not preserved except int64
    n = 10
    d = 2
    degree = 2

    inputs = generate_random_inputs(n, d).to(dtype)
    predicted = PolynomialFeatures(degree).fit(inputs).transform(inputs)

    assert predicted.dtype == dtype


def test_runtime_error_if_not_fitted():
    inputs = generate_random_inputs(30, 1)

    with pytest.raises(RuntimeError):
        PolynomialFeatures().transform(inputs)


def test_value_error():
    X = generate_random_inputs(15, 5)
    Y = generate_random_inputs(20, 3)

    with pytest.raises(ValueError, match="shape"):
        PolynomialFeatures().fit(X).transform(Y)


@pytest.mark.cuda
def test_cuda_matches_cpu():
    inputs = generate_random_inputs(10, 2)
    predicted = PolynomialFeatures(2).fit(inputs).transform(inputs)

    device = torch.device("cuda")
    inputs = inputs.to(device)
    predicted_cuda = PolynomialFeatures(2).fit(inputs).transform(inputs)

    assert predicted_cuda.device.type == "cuda"

    assert torch.allclose(predicted, predicted_cuda.cpu())
