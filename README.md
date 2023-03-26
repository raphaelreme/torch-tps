# torch-tps

[![Lint and Test](https://github.com/raphaelreme/torch-tps/actions/workflows/tests.yml/badge.svg)](https://github.com/raphaelreme/torch-tps/actions/workflows/tests.yml)

Implementation of Thin Plate Spline.
(See numpy implementation with [thin-plate-spline](https://github.com/raphaelreme/tps) library)


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

## Examples

We provide different examples in the `example` folder. (From interpolation, to multidimensional cases and image warping).


### Image warping

The elastic deformation of TPS can be used for image warping. Here is an example of tps to increase/decrease the size of the center of the image or using random control points:

![Input Image](example/images/dog_with_bbox.png)![Increased Image](example/images/increase_warped_dog.png)![Decreased Image](example/images/decrease_warped_dog.png)![Warped Image](example/images/random_warped_dog.png)

Have a look at `example/image_warping.py`.

## Build and Deploy

```bash
$ python -m build
$ python -m twine upload dist/*
```
