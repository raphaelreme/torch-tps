"""Thin Plate Spline in PyTorch.

PyTorch implementation of the generalized Polyharmonic Spline interpolation
(also known as Thin Plate Spline in 2D).
It learns a smooth elastic mapping between two Euclidean spaces with support for:

* Arbitrary input and output dimensions
* Arbitrary spline order `k`
* Optional regularization
* Supports CPU and GPU parallelization

Useful for interpolation, deformation fields, and smooth non-linear regression.

For a NumPy implementation, see [tps](https://github.com/raphaelreme/tps).

This implementation is much faster than the NumPy one, thanks to the cpu //.
Using gpu seems not to be much faster for fitting (linear system solving),
but is much faster to transform (as this is simply a matrix multiplication).

Getting started:
---------------

```python
import torch
from torch_tps import ThinPlateSpline

# Control points
X_train = torch.random.normal(0, 1, (800, 3))  # 800 points in R^3
Y_train = torch.random.normal(0, 1, (800, 2))  # Values for each point (800 values in R^2)

# New source points to interpolate
X_test = torch.random.normal(0, 1, (3000, 3))

# Initialize spline model (Regularization is controlled with alpha parameter)
tps = ThinPlateSpline(alpha=0.5)  # Use device="cuda" to switch to gpu

# Fit spline from control points
tps.fit(X_train, Y_train)

# Interpolate new points
Y_test = tps.transform(X_test)
```

Please refer to the ![official documentation](https://github.com/raphaelreme/torch-tps)

License
-------

MIT License
"""

import importlib.metadata

from .polynomial_transform import PolynomialFeatures
from .thin_plate_spline import ThinPlateSpline

__all__ = ["PolynomialFeatures", "ThinPlateSpline"]
__version__ = importlib.metadata.version("torch_tps")
