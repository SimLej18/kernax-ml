from __future__ import annotations

import jax.numpy as jnp
from equinox import filter_jit
from jax import Array

from ..AbstractMean import AbstractMean, StaticAbstractMean


class StaticZeroMean(StaticAbstractMean):
	@classmethod
	@filter_jit
	def scalar_mean(cls, mean: AbstractMean, x: Array) -> Array:
		return jnp.array(0.0)


class ZeroMean(AbstractMean):
	static_class = StaticZeroMean