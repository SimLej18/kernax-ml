from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from equinox import filter_jit
from jax import Array

from .WrapperModule import WrapperModule


class LogModule(WrapperModule):
	"""Module that applies the logarithm function to the output of another module."""

	@filter_jit
	def __call__(self, x1: Array, x2: Optional[Array] = None) -> Array:
		if x2 is None:
			return jnp.log(self.inner(x1))
		return jnp.log(self.inner(x1, x2))

	def __str__(self):
		return f"Log({self.inner})"