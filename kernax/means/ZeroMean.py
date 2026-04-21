from __future__ import annotations
import jax.numpy as jnp
from jax import Array
from ..AbstractMean import AbstractMean


class ZeroMean(AbstractMean):
	def scalar_mean(self, x: Array) -> Array:
		return jnp.array(0.0)

	def replace(self, **kwargs) -> ZeroMean:
		return self  # Nothing to mutate
