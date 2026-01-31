from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional, Type

import equinox as eqx
import jax.numpy as jnp
from equinox import filter_jit
from jax import Array, vmap
from jax.lax import cond

from .utils import format_jax_array

if TYPE_CHECKING:
	pass


class AbstractKernel(eqx.Module):
	static_class: ClassVar[Optional[Type[StaticAbstractKernel]]] = (
		None  # Must be defined in sub-class
	)

	def __init__(self):
		"""
		Initialize the kernel and mark that a kernel has been instantiated.

		This locks the parameter_transform config setting to prevent inconsistencies
		with JIT-compiled code.
		"""
		# Import here to avoid circular dependency
		from .config import config

		# Mark that kernels have been instantiated (locks parameter_transform)
		config._mark_kernel_instantiated()

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
			return self.static_class.pairwise_cov_if_not_nan(self, x1, x2)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 1:
			return self.static_class.cross_cov_vector_if_not_nan(self, x1, x2)
		elif jnp.ndim(x1) == 1 and jnp.ndim(x2) == 2:
			return self.static_class.cross_cov_vector_if_not_nan(self, x2, x1)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 2:
			return self.static_class.cross_cov_matrix(self, x1, x2)
		else:
			raise ValueError(
				f"Invalid input dimensions: x1 has shape {x1.shape}, x2 has shape {x2.shape}. "
				"Expected scalar, 1D or 2D arrays as inputs."
			)

	def __add__(self, other):
		from kernax.operators import SumKernel

		return SumKernel(self, other)

	def __radd__(self, other):
		from kernax.operators import SumKernel

		return SumKernel(other, self)

	def __sub__(self, other):
		from kernax.operators import SumKernel
		from kernax.wrappers import NegKernel

		return SumKernel(self, NegKernel(other))

	def __rsub__(self, other):
		from kernax.operators import SumKernel
		from kernax.wrappers import NegKernel

		return SumKernel(other, NegKernel(self))

	def __neg__(self):
		from kernax.wrappers import NegKernel

		return NegKernel(self)

	def __mul__(self, other):
		from kernax.operators import ProductKernel

		return ProductKernel(self, other)

	def __rmul__(self, other):
		from kernax.operators import ProductKernel

		return ProductKernel(other, self)

	def __str__(self):
		# FIXME: do not call `format_jax_array` when the pytree is filled with non-float values
		#  For example, try to print the `batch_in_axes` property of a BatchKernel
		# Print parameters, aka elements of __dict__ that are jax arrays
		return f"{self.__class__.__name__}({
			', '.join(
				[
					f'{key}={format_jax_array(value)}'
					for key, value in self.__dict__.items()
					if isinstance(value, Array)
				]
			)
		})"


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
		Returns NaN if either x1 or x2 is NaN, otherwise calls the compute_scalar method.

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
		Returns an array of NaN if scalar is NaN, otherwise calls the compute_vector method.

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
