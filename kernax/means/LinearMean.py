from __future__ import annotations
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from ..AbstractMean import AbstractMean


class LinearMean(AbstractMean):
	slope: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, slope: float | Array = 0.0):
		self.slope = jnp.asarray(slope)

	def scalar_mean(self, x: Array) -> Array:
		return jnp.sum(self.slope * x)

	def replace(self, slope: None | float | Array = None, **kwargs) -> LinearMean:
		if slope is None:
			return self
		return eqx.tree_at(
			lambda m: m.slope,
			self,
			jnp.broadcast_to(jnp.asarray(slope), self.slope.shape)
		)
