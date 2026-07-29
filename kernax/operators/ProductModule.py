from __future__ import annotations
from jax import Array
from .AbstractOperatorModule import AbstractOperatorModule


class ProductModule(AbstractOperatorModule):
	"""Operator module that multiplies the outputs of two sub-modules."""

	def __call__(self, x1: Array, x2: Array | None = None, **kwargs) -> Array:
		if x2 is None:
			return self.left(x1, **kwargs) * self.right(x1, **kwargs)
		return self.left(x1, x2, **kwargs) * self.right(x1, x2, **kwargs)

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

	def factors(self, x1: Array, x2: Array | None = None, **kwargs) -> tuple[tuple[Array, Array], ...]:
		"""
		Kronecker terms of a multi-output kernel: ``((B_1, K_1), ...)`` for ``sum_q B_q (x) K_q``.

		Uses ``(A (x) B) o (C (x) D) = (A o C) (x) (B o D)`` (``o`` being the Hadamard
		product), so the product of two sums of Kronecker terms is the Hadamard product of
		every pair of terms. Only defined when both operands expose `factors`.
		"""
		left, right = self.left.factors(x1, x2, **kwargs), self.right.factors(x1, x2, **kwargs)
		return tuple((bl * br, kl * kr) for bl, kl in left for br, kr in right)
