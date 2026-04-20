from __future__ import annotations
from equinox import filter_jit
import jax.numpy as jnp
from jax import Array
from ..AbstractMean import AbstractMean


class ZeroMean(AbstractMean):
	@filter_jit
	def scalar_mean(self, x: Array) -> Array:
		return jnp.array(0.0)

	def replace(self, **kwargs) -> ZeroMean:
		return self  # Nothing to mutate
