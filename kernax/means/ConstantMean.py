from __future__ import annotations
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from ..AbstractMean import AbstractMean


class ConstantMean(AbstractMean):
	constant: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, constant: float | Array = 0.0):
		self.constant = jnp.asarray(constant)

	def scalar_mean(self, x: Array) -> Array:
		return self.constant

	def replace(self, constant: None | float | Array = None, **kwargs) -> ConstantMean:
		if constant is None:
			return self
		return eqx.tree_at(
			lambda m: m.constant,
			self,
			jnp.broadcast_to(jnp.asarray(constant), self.constant.shape)
		)
