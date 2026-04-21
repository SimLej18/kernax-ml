from __future__ import annotations
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from ..AbstractMean import AbstractMean


class AffineMean(AbstractMean):
	slope: Array = eqx.field(converter=jnp.asarray)
	intercept: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, slope: float | Array = 0.0, intercept: float | Array = 0.0):
		self.slope = jnp.asarray(slope)
		self.intercept = jnp.asarray(intercept)

	def scalar_mean(self, x: Array) -> Array:
		return jnp.sum(self.slope * x) + self.intercept

	def replace(self,
	            slope: None | float | Array = None,
	            intercept: None | float | Array = None,
	            **kwargs) -> AffineMean:
		new_mean = self

		if slope is not None:
			new_mean = eqx.tree_at(
				lambda m: m.slope,
				new_mean,
				jnp.broadcast_to(jnp.asarray(slope), self.slope.shape)
			)

		if intercept is not None:
			new_mean = eqx.tree_at(
				lambda m: m.intercept,
				new_mean,
				jnp.broadcast_to(jnp.asarray(intercept), self.intercept.shape)
			)

		return new_mean
