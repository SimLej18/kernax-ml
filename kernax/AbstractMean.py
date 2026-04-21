from __future__ import annotations
from abc import abstractmethod
import jax.numpy as jnp
from jax import Array, vmap
from .module import AbstractModule


class AbstractMean(AbstractModule):
	@abstractmethod
	def scalar_mean(self, x: Array) -> Array:
		"""Compute the mean value for a single input vector."""
		raise NotImplementedError

	def __call__(self, x: Array, *args, **kwargs) -> Array:
		x = jnp.atleast_1d(x)

		if jnp.ndim(x) == 1:
			return self.scalar_mean(x)
		elif jnp.ndim(x) == 2:
			return vmap(self.scalar_mean)(x)
		else:
			raise ValueError(
				f"Invalid input dimensions: x has shape {x.shape}. "
				"Expected 1D or 2D arrays as inputs."
			)

	@abstractmethod
	def replace(self, x: Array) -> Array:
		raise NotImplementedError
