import pytest
import torch

from torch_tps import ThinPlateSpline


def test_interpolates_training_points_alpha0():
    rng = torch.Generator().manual_seed(0)
    X = torch.randn((50, 3), generator=rng)
    Y = torch.randn((50, 2), generator=rng)

    tps = ThinPlateSpline(alpha=0.0, order=2)
    tps.fit(X, Y)
    Y_pred = tps.transform(X)

    tol = 1e-4
    assert (Y_pred - Y).abs().max() < tol


def test_regularization_relaxes_interpolation():
    rng = torch.Generator().manual_seed(0)
    X = torch.randn((20, 3), generator=rng)
    Y = torch.randn((20, 2), generator=rng)

    tps = ThinPlateSpline(alpha=0.05, order=2).fit(X, Y)

    error = torch.max(torch.abs(tps.transform(X) - Y))

    tol = 1e-2
    assert error > tol  # should generally not be exact anymore


def test_permutation_invariance():
    rng = torch.Generator().manual_seed(0)
    X = torch.randn((80, 4), generator=rng)
    Y = torch.randn((80,), generator=rng)
    X_test = torch.randn((30, 4), generator=rng)
    perm = torch.randperm(len(X), generator=rng)

    tps = ThinPlateSpline(alpha=0.05, order=2).fit(X, Y)
    tps_perm = ThinPlateSpline(alpha=0.05, order=2).fit(X[perm], Y[perm])

    Y_pred = tps.transform(X_test)
    Y_perm = tps_perm.transform(X_test)

    tol = 1e-3  # NOTE: torch implem is quite sensitive
    assert (Y_pred - Y_perm).abs().max() < tol


def test_affine_reproduction_order_2():
    rng = torch.Generator().manual_seed(0)
    X = torch.randn((100, 5), generator=rng)
    A = torch.randn((5, 3), generator=rng)
    b = torch.randn((3,), generator=rng)
    Y = X @ A + b

    X_test = torch.randn((40, 5), generator=rng)
    Y_true = X_test @ A + b

    # For order=2, polynomial degree is 1: any affine function should be reproduced exactly.
    tps = ThinPlateSpline(alpha=0.0, order=2).fit(X, Y)
    Y_pred = tps.transform(X_test)

    tol = 1e-5  # NOTE: torch implem is quite sensitive
    assert (Y_pred - Y_true).abs().max() < tol


def test_known_3d_function_regression_sanity():
    def f(X):
        return (torch.sin(X[:, 0]) + torch.cos(X[:, 1]) + X[:, 2] ** 2)[:, None]

    rng = torch.Generator().manual_seed(0)
    Xc = torch.rand((500, 3), generator=rng) * 8 - 4
    Yc = f(Xc)

    Xt = torch.rand((50, 3), generator=rng) * 6 - 3
    Yt = f(Xt)

    tps = ThinPlateSpline(alpha=0.1, order=2).fit(Xc, Yc)
    Y_pred = tps.transform(Xt)

    tol = 1e-2
    mse = torch.mean((Y_pred - Yt) ** 2)
    assert mse < tol


def test_duplicate_fails_without_regularization():
    # TPS becomes singular with duplicates when alpha=0.
    X = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],  # duplicate
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rng = torch.Generator().manual_seed(0)
    Y = torch.randn((4, 1), generator=rng)

    tps = ThinPlateSpline(alpha=0.0, order=2)

    with pytest.raises(Exception, match="singular"):
        tps.fit(X, Y)


def test_duplicate_works_with_regularization():
    X = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],  # duplicate
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rng = torch.Generator().manual_seed(0)
    Y = torch.randn((4, 1), generator=rng)

    tps = ThinPlateSpline(alpha=1e-3, order=2)
    tps.fit(X, Y)

    assert torch.all(torch.isfinite(tps.parameters)), "Parameters should be finite with regularization"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_conserved(dtype: torch.dtype):
    rng = torch.Generator().manual_seed(0)
    X = torch.randn((20, 1), generator=rng).to(dtype)
    Y = torch.randn((20, 5), generator=rng).to(dtype)
    X_test = torch.randn((30, 1), generator=rng).to(dtype)

    predicted = ThinPlateSpline().fit(X, Y).transform(X_test)
    assert predicted.dtype == dtype


def test_runtime_error_if_not_fitted():
    rng = torch.Generator().manual_seed(0)
    X_test = torch.randn((30, 1), generator=rng)

    with pytest.raises(RuntimeError):
        ThinPlateSpline().transform(X_test)


def test_value_error():
    rng = torch.Generator().manual_seed(0)
    X = torch.randn((15, 1), generator=rng)
    Y = torch.randn((20, 5), generator=rng)

    with pytest.raises(ValueError, match="same number of points"):
        ThinPlateSpline().fit(X, Y)

    X = torch.randn((20, 5, 5), generator=rng)
    Y = torch.randn((20, 5), generator=rng)

    with pytest.raises(ValueError, match="2d"):
        ThinPlateSpline().fit(X, Y)

    X = torch.randn((15, 5), generator=rng)
    Y = torch.randn((20, 3), generator=rng)

    with pytest.raises(ValueError, match="features"):
        ThinPlateSpline().fit(X, X).transform(Y)


@pytest.mark.cuda
def test_cuda_matches_cpu():
    rng = torch.Generator().manual_seed(0)
    X = torch.randn((20, 4), generator=rng).to("cuda")  # Switch one to cuda
    Y = torch.randn((20, 4), generator=rng)
    X_test = torch.randn((10, 4), generator=rng)

    tps = ThinPlateSpline(alpha=1e-3, order=2, device="cpu")
    Y_test = tps.fit(X, Y).transform(X_test)  # Inputs are converted to the tps device

    assert Y_test.device.type == "cpu"

    tps = ThinPlateSpline(alpha=1e-3, order=2, device="cuda")
    Y_test_cuda = tps.fit(X, Y).transform(X_test)  # Inputs are converted to the tps device

    assert Y_test_cuda.device.type == "cuda"

    tol = 1e-5  # NOTE: torch implem is quite sensitive
    assert (Y_test - Y_test_cuda.cpu()).abs().max() < tol
