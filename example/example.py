import time

import matplotlib.pyplot as plt  # type: ignore
import torch
import tqdm  # type: ignore

from torch_tps import ThinPlateSpline


def compute_time(f, n, *args, **kwargs):
    elapsed_time = 0.0
    for _ in tqdm.trange(n):
        t = time.time()
        f(*args, **kwargs)
        elapsed_time += time.time() - t

    return elapsed_time / n


def main_timed():
    """Check time execution

    Also shows d_s != d_t, and can be switch to cuda to see run time gains
    """
    device = torch.device("cpu")

    d = 3
    v = 2
    n = 1500
    m = 5000

    X_train = torch.normal(0, 2, (n, d), device=device)
    Y_train = torch.normal(0, 1, (n, v), device=device)
    X_test = torch.normal(0, 1, (m, d), device=device)

    tps = ThinPlateSpline(0.0, device=device)

    fit_time = compute_time(tps.fit, 100, X_train, Y_train)
    transform_time = compute_time(tps.transform, 100, X_test)

    print(f"Control point number: {n}")
    print(f"Transformed point number: {m}")
    print(f"Source and target space dimension: {d} -> {v}")
    print(f"Fit avg time: {fit_time}")
    print(f"Transform avg time: {transform_time}")


def main_interpolation():
    """Interpolates a function

    Inspired from:
    https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
    """

    def f(x):
        return x * torch.sin(x)

    # whole range we want to plot
    x_plot = torch.linspace(-1, 11, 100)
    x_train = torch.linspace(0, 10, 100)
    torch.manual_seed(0)
    idx = torch.multinomial(torch.ones(x_train.shape), 20, replacement=False)
    x_train = x_train[idx]
    y_train = f(x_train)

    # Fit the model and transform
    tps = ThinPlateSpline(0.5)
    y_pred = tps.fit(x_train, f(x_train)).transform(x_plot)[:, 0]

    # Plot everything
    plt.plot(x_plot.numpy(), f(x_plot).numpy(), label="ground truth")
    plt.plot(x_plot.numpy(), y_pred.numpy())
    plt.scatter(x_train.numpy(), y_train.numpy(), label="training points")
    plt.show()


def main_surface_mapping():
    """Maps two surfaces

    Inspired from https://github.com/tzing/tps-deformation
    """
    samp = torch.linspace(-2, 2, 4)
    xx, yy = torch.meshgrid(samp, samp, indexing="ij")

    # make source surface, get uniformed distributed control points
    source_xy = torch.stack([xx, yy], dim=2).reshape(-1, 2)

    # make deformed surface
    yy = yy.clone()
    yy[0] *= 2
    yy[3] *= 2
    deform_xy = torch.stack([xx, yy], dim=2).reshape(-1, 2)

    tps = ThinPlateSpline(0.5)

    # Fit the surfaces
    tps.fit(source_xy, deform_xy)

    # make other points a left-bottom to upper-right line on source surface
    samp2 = torch.linspace(-1.8, 1.8, 100)
    test_xy = torch.tile(samp2, [2, 1]).T

    # get transformed points
    transformed_xy = tps.transform(test_xy)

    plt.figure()
    plt.scatter(source_xy[:, 0].numpy(), source_xy[:, 1].numpy())
    plt.plot(test_xy[:, 0].numpy(), test_xy[:, 1].numpy(), c="orange")

    plt.figure()
    plt.scatter(deform_xy[:, 0].numpy(), deform_xy[:, 1].numpy())
    plt.plot(transformed_xy[:, 0].numpy(), transformed_xy[:, 1].numpy(), c="orange")

    plt.show()


if __name__ == "__main__":
    main_surface_mapping()
    main_interpolation()
    main_timed()
