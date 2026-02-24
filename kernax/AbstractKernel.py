from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional, Type

import equinox as eqx
import jax.numpy as jnp
from equinox import filter_jit
from jax import Array, vmap
from jax.lax import cond

from .engines import ComputationEngine, DenseEngine
from .utils import format_jax_array
from .module import AbstractModule

if TYPE_CHECKING:
	pass


class AbstractKernel(AbstractModule):
	static_class: ClassVar[Optional[Type[StaticAbstractKernel]]] = None
	computation_engine: Type[ComputationEngine] = eqx.field(
		static=True, default=DenseEngine, kw_only=True
	)

	def __init__(self, computation_engine=DenseEngine, **kwargs):
		"""
		Initialize the kernel.

		Args:
			computation_engine: The computation engine to use for covariance calculations
			**kwargs: Additional keyword arguments (for subclass compatibility)
		"""
		super().__init__(**kwargs)

		# Set the computation engine
		self.computation_engine = computation_engine


	@filter_jit
	def __call__(self, x1: Array, x2: Optional[Array] = None) -> Array:
		# If no x2 is provided, we compute the covariance between x1 and itself
		if x2 is None:
			x2 = x1

		# Turn scalar inputs into vectors
		x1, x2 = jnp.atleast_1d(x1), jnp.atleast_1d(x2)

		# Ensure static_class is not None
		assert self.static_class is not None, "static_class must be defined in subclass"

		# Call the appropriate method
		if jnp.ndim(x1) == 1 and jnp.ndim(x2) == 1:
			return self.computation_engine.pairwise_cov_if_not_nan(self, x1, x2)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 1:
			return self.computation_engine.cross_cov_vector_if_not_nan(self, x1, x2)
		elif jnp.ndim(x1) == 1 and jnp.ndim(x2) == 2:
			return self.computation_engine.cross_cov_vector_if_not_nan(self, x2, x1)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 2:
			return self.computation_engine.cross_cov_matrix(self, x1, x2)
		else:
			raise ValueError(
				f"Invalid input dimensions: x1 has shape {x1.shape}, x2 has shape {x2.shape}. "
				"Expected scalar, 1D or 2D arrays as inputs."
			)


class StaticAbstractKernel:
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return jnp.array(jnp.nan)  # To be overwritten in subclasses

	@classmethod
	@filter_jit
	def pairwise_cov_if_not_nan(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Returns NaN if either x1 or x2 is NaN, otherwise calls the pairwise_cov method.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return cond(  # type: ignore[no-any-return]
			jnp.any(jnp.isnan(x1) | jnp.isnan(x2)),
			lambda _: jnp.nan,
			lambda _: cls.pairwise_cov(kern, x1, x2),
			None,
		)

	@classmethod
	@filter_jit
	def cross_cov_vector(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel cross covariance values between an array of vectors (matrix) and a vector.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: scalar array
		:return: vector array (N, )
		"""
		return vmap(lambda x: cls.pairwise_cov_if_not_nan(kern, x, x2), in_axes=0)(x1)

	@classmethod
	@filter_jit
	def cross_cov_vector_if_not_nan(
		cls, kern: AbstractKernel, x1: Array, x2: Array, **kwargs
	) -> Array:
		"""
		Returns an array of NaN if scalar is NaN, otherwise calls the cross_cov_vector method.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return cond(  # type: ignore[no-any-return]
			jnp.any(jnp.isnan(x2)),
			lambda _: jnp.full(len(x1), jnp.nan),
			lambda _: cls.cross_cov_vector(kern, x1, x2),
			None,
		)

	@classmethod
	@filter_jit
	def cross_cov_matrix(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel covariance matrix between two vector arrays.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:return: matrix array (N, M)
		"""
		return vmap(lambda x: cls.cross_cov_vector_if_not_nan(kern, x2, x), in_axes=0)(x1)
