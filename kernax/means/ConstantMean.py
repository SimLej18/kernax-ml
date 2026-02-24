from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from equinox import filter_jit
from jax import Array

from ..AbstractMean import AbstractMean, StaticAbstractMean


class StaticConstantMean(StaticAbstractMean):
	@classmethod
	@filter_jit
	def scalar_mean(cls, mean: AbstractMean, x: Array) -> Array:
		return mean.constant


class ConstantMean(AbstractMean):
	constant: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticConstantMean

	def __init__(self, constant=0.0, **kwargs):
		super().__init__(**kwargs)
		self.constant = jnp.asarray(constant)