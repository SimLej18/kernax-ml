from __future__ import annotations

from typing import Optional

from equinox import filter_jit
from jax import Array

from .WrapperModule import WrapperModule


class NegModule(WrapperModule):
	@filter_jit
	def __call__(self, x1: Array, x2: Optional[Array] = None) -> Array:
		if x2 is None:
			return -self.inner(x1)
		return -self.inner(x1, x2)

	def __str__(self):
		return f"- {self.inner}"