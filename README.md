# torch-tps

[![Lint and Test](https://github.com/raphaelreme/torch-tps/actions/workflows/tests.yml/badge.svg)](https://github.com/raphaelreme/torch-tps/actions/workflows/tests.yml)

Implementation of Thin Plate Spline.
(See numpy implementation with thin-plate-spline library)


## Install

### Pip

```bash
$ pip install torch-tps
```

### Conda

Not yet available


## Getting started

```python

import torch
from tps import ThinPlateSpline

# Some data
X_c = torch.normal(0, 1, (800, 3))
X_t = torch.normal(0, 2, (800, 2))
X = torch.normal(0, 1, (300, 3))

# Create the tps object
tps = ThinPlateSpline(alpha=0.0)  # 0 Regularization

# Fit the control and target points
tps.fit(X_c, X_t)

# Transform new points
Y = tps.transform(X)
```

Also have a look at `example.py`


## Build and Deploy

```bash
$ python -m build
$ python -m twine upload dist/*
```
