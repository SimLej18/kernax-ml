from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from equinox import filter_jit
from jax import Array

from ..AbstractMean import AbstractMean, StaticAbstractMean


class StaticLinearMean(StaticAbstractMean):
	@classmethod
	@filter_jit
	def scalar_mean(cls, mean: AbstractMean, x: Array) -> Array:
		return jnp.sum(mean.slope * x)


class LinearMean(AbstractMean):
	slope: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticLinearMean

	def __init__(self, slope=0.0, **kwargs):
		super().__init__(**kwargs)
		self.slope = jnp.asarray(slope)