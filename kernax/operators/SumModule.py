from __future__ import annotations

from typing import Generic

from jax import Array

from ..module import ModuleT
from .AbstractOperatorModule import AbstractOperatorModule


class SumModule(AbstractOperatorModule[ModuleT], Generic[ModuleT]):
	"""Operator module that sums the outputs of two sub-modules."""

	def __call__(self, x1: Array, x2: Array | None = None, **kwargs) -> Array:
		if x2 is None:
			return self.left(x1, **kwargs) + self.right(x1, **kwargs)
		return self.left(x1, x2, **kwargs) + self.right(x1, x2, **kwargs)

	def __str__(self):
		if self.right.__class__.__name__ == "NegModule":
			return f"{self.left} - {self.right.inner}"  # type: ignore[attr-defined]
		return f"{self.left} + {self.right}"

	def spectral_density(self, w):
		# The Fourier transform is linear, so the spectral density of a sum is the sum of
		# the spectral densities.
		return self.left.spectral_density(w) + self.right.spectral_density(w)

	def factors(self, x1: Array, x2: Array | None = None, **kwargs) -> tuple[tuple[Array, Array], ...]:
		"""
		Kronecker terms of a multi-output kernel: ``((B_1, K_1), ...)`` for ``sum_q B_q (x) K_q``.

		A sum of multi-output kernels simply concatenates the terms of its operands, which is
		the LMC form. Only defined when both operands expose `factors`, i.e. when both are
		multi-output kernels over shared inputs.
		"""
		return self.left.factors(x1, x2, **kwargs) + self.right.factors(x1, x2, **kwargs)
