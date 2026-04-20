from __future__ import annotations
from equinox import filter_jit
from jax import Array
from .AbstractOperatorModule import AbstractOperatorModule


class SumModule(AbstractOperatorModule):
	"""Operator module that sums the outputs of two sub-modules."""

	@filter_jit
	def __call__(self, x1: Array, x2: Array | None = None) -> Array:
		if x2 is None:
			return self.left(x1) + self.right(x1)
		return self.left(x1, x2) + self.right(x1, x2)

	def __str__(self):
		if self.right.__class__.__name__ == "NegModule":
			return f"{self.left} - {self.right.inner}"  # type: ignore[attr-defined]
		return f"{self.left} + {self.right}"
