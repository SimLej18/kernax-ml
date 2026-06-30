from __future__ import annotations
from jax import Array
from .AbstractOperatorModule import AbstractOperatorModule


class ProductModule(AbstractOperatorModule):
	"""Operator module that multiplies the outputs of two sub-modules."""

	def __call__(self, x1: Array, x2: Array | None = None) -> Array:
		if x2 is None:
			return self.left(x1) * self.right(x1)
		return self.left(x1, x2) * self.right(x1, x2)

	def __str__(self):
		if self.right.__class__.__name__ == "NegModule":
			return f"{self.left} * ({self.right})"
		return f"{self.left} * {self.right}"

	def spectral_density(self, w):
		from kernax import VarianceKernel

		# Will only work if either left or right kernel is a VarianceKernel, or a BatchModule of
		# a VarianceKernel
		if not (isinstance(self.left, VarianceKernel) or isinstance(self.right, VarianceKernel)):
			raise NotImplementedError(
				"spectral_density only supported when one operand is a VarianceKernel."
			)
		return self.left.spectral_density(w) * self.right.spectral_density(w)
