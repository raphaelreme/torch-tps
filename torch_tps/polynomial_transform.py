"""Polynomial transform adapted from scikit-learn"""

from itertools import chain, combinations_with_replacement
from typing import Iterable, Tuple

import torch


class PolynomialFeatures:
    """Generate polynomial features

    For each feature vector, it generates polynomial features of degree d consisting
    of all polynomial combinations of the features with degree less than or equal to
    the specified degree.

    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    With degree 1, it simply transform to homogenous coordinates (adding a constant 1 to the features)

    See PolynomialFeatures from scikit-learn for a complete documentation and implementation.

    Contrary to sklearn implementation, we computes the full exponents matrix (d_p x d)
    And exploits GPU a bit inefficiently to compute X_p (but much much faster than for loop with sparse memory access)
    """

    def __init__(self, degree=1):
        self.degree = degree
        self._fitted = False
        self.exponents = torch.tensor([], dtype=torch.int32)

    @staticmethod
    def _combinations(n_features: int, degree: int) -> Iterable[Tuple[int, ...]]:
        return chain.from_iterable(combinations_with_replacement(range(n_features), i) for i in range(degree + 1))

    def fit(self, X: torch.Tensor):
        """Compute number of output features."""
        _, n_features = X.shape
        combinations = list(self._combinations(n_features, self.degree))

        self.exponents = torch.zeros(len(combinations), n_features, dtype=torch.int32, device=X.device)
        for i, combination in enumerate(combinations):
            for j in combination:
                self.exponents[i, j] += 1

        self._fitted = True
        return self

    def transform(self, X: torch.Tensor):
        """Transform features to polynomial features

        Args:
            X (torch.Tensor): Features for multiple samples
                Shape: (n_samples, n_features)

        Returns
            torch.Tensor: Polynomial features
                Shape: (n_samples, n_output_features)
        """
        assert self._fitted, "Please first fit the PolynomialFeatures"

        if X.shape[1] != self.exponents.shape[1]:
            raise ValueError("X shape does not match training shape")

        # Could reduce memory footprint and computations by using a dedicated kernel
        # But fast enough (usually d < d_p << n and therefore this is negligible before cdist computations)
        return (X[:, None] ** self.exponents).prod(dim=-1)
