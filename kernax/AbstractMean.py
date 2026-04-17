from __future__ import annotations
from abc import abstractmethod
import jax.numpy as jnp
from jax import Array
from equinox import filter_jit
from .module import AbstractModule


class AbstractMean(AbstractModule):
	@filter_jit
	def __call__(self, x: Array, *args, **kwargs) -> Array:
		x = jnp.atleast_1d(x)

		if jnp.ndim(x) == 1:
			return self.scalar_mean(self, x, *args, **kwargs)
		elif jnp.ndim(x) == 2:
			return self.vector_mean(self, x, *args, **kwargs)
		else:
			raise ValueError(
				f"Invalid input dimensions: x has shape {x.shape}. "
				"Expected 1D or 2D arrays as inputs."
			)

	@abstractmethod
	def scalar_mean(cls, mean: AbstractMean, x: Array) -> Array:
		"""
		Compute the mean value for a single input vector.

		:param mean: mean instance containing the parameters
		:param x: 1D array
		:return: scalar array
		"""
		raise NotImplementedError

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
