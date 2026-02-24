from __future__ import annotations

from typing import ClassVar, Optional, Type

import jax.numpy as jnp
from equinox import filter_jit
from jax import Array, vmap

from .module import AbstractModule, StaticAbstractModule


class AbstractMean(AbstractModule):
	static_class: ClassVar[Optional[Type[StaticAbstractMean]]] = None

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	@filter_jit
	def __call__(self, x: Array) -> Array:
		x = jnp.atleast_1d(x)

		assert self.static_class is not None, "static_class must be defined in subclass"

		if jnp.ndim(x) == 1:
			return self.static_class.scalar_mean(self, x)
		elif jnp.ndim(x) == 2:
			return self.static_class.vector_mean(self, x)
		else:
			raise ValueError(
				f"Invalid input dimensions: x has shape {x.shape}. "
				"Expected 1D or 2D arrays as inputs."
			)


class StaticAbstractMean(StaticAbstractModule):
	@classmethod
	@filter_jit
	def scalar_mean(cls, mean: AbstractMean, x: Array) -> Array:
		"""
		Compute the mean value for a single input vector.

		:param mean: mean instance containing the parameters
		:param x: 1D array
		:return: scalar array
		"""
		return jnp.array(jnp.nan)  # To be overwritten in subclasses

	@classmethod
	@filter_jit
	def vector_mean(cls, mean: AbstractMean, x: Array) -> Array:
		"""
		Compute the mean for a batch of input vectors.

		:param mean: mean instance containing the parameters
		:param x: 2D array (N, D)
		:return: 1D array (N,)
		"""
		return vmap(cls.scalar_mean, in_axes=(None, 0))(mean, x)