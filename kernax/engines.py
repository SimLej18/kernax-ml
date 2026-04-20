from __future__ import annotations
from abc import abstractmethod
import jax.numpy as jnp
import jax.lax as jlx
from jax import Array, vmap
import equinox as eqx
from equinox import filter_jit


class AbstractEngine(eqx.Module):
	def __new__(cls, *args, **kwargs):
		raise TypeError("Engines are static, they should not be instantiated")

	@classmethod
	@abstractmethod
	def __call__(cls, module, x1, x2, *args, **kwargs) -> Array:
		raise NotImplementedError


class DenseEngine(AbstractEngine):
	""" Computes the dense cross-covariance matrix, sometimes called the Gram matrix of a kernel"""
	@staticmethod
	@filter_jit
	def __call__(module: AbstractModule, x1: Array, x2: Array, *args, **kwargs) -> Array:
		# Turn scalar inputs into vectors
		x1, x2 = jnp.atleast_1d(x1), jnp.atleast_1d(x2)

		# Call the appropriate method
		if x1.ndim == 1 and x2.ndim == 1:
			return DenseEngine.pairwise(module, x1, x2)
		elif x1.ndim == 2 and x2.ndim == 1:
			return DenseEngine.cross_cov_vector(module, x2, x1)
		elif x1.ndim == 1 and x2.ndim == 2:
			return DenseEngine.cross_cov_vector(module, x1, x2)
		elif x1.ndim == 2 and x2.ndim == 2:
			return DenseEngine.cross_cov_matrix(module, x1, x2)
		else:
			raise ValueError(
				f"Invalid input dimensions: x1 has shape {x1.shape}, x2 has shape {x2.shape}. "
				"Expected scalar, 1D or 2D arrays as inputs."
			)

	@staticmethod
	@filter_jit
	def pairwise(module: AbstractModule, x1: Array, x2: Array):
		return module.pairwise(x1, x2)

	@staticmethod
	@filter_jit
	def cross_cov_vector(module: AbstractModule, x1: Array, x2: Array) -> Array:
		return vmap(DenseEngine.pairwise, in_axes=(None, None, 0))(module, x1, x2)

	@staticmethod
	@filter_jit
	def cross_cov_matrix(module: AbstractModule, x1: Array, x2: Array) -> Array:
		return vmap(DenseEngine.cross_cov_vector, in_axes=(None, 0, None))(module, x1, x2)


class NaNDenseEngine(AbstractEngine):
	"""
	Engine optimised to shortcut some computations when there are NaNs in some inputs.

	N.b: speedups are not always automatic, especially when JITing computations and running on GPUs.
	Using a DenseEngine (or a NoJitDenseEngine when JITing is unavailable) is often as fast or even faster.
	"""
	@staticmethod
	@filter_jit
	def __call__(module: AbstractModule, x1: Array, x2: Array, *args, **kwargs) -> Array:
		# Turn scalar inputs into vectors
		x1, x2 = jnp.atleast_1d(x1), jnp.atleast_1d(x2)

		# Call the appropriate method
		if jnp.ndim(x1) == 1 and jnp.ndim(x2) == 1:
			return NaNDenseEngine.pairwise_if_not_nan(module, x1, x2)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 1:
			return NaNDenseEngine.cross_cov_vector_if_not_nan(module, x2, x1)
		elif jnp.ndim(x1) == 1 and jnp.ndim(x2) == 2:
			return NaNDenseEngine.cross_cov_vector_if_not_nan(module, x1, x2)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 2:
			return NaNDenseEngine.cross_cov_matrix(module, x1, x2)
		else:
			raise ValueError(
				f"Invalid input dimensions: x1 has shape {x1.shape}, x2 has shape {x2.shape}. "
				"Expected scalar, 1D or 2D arrays as inputs."
			)

	@staticmethod
	@filter_jit
	def pairwise_if_not_nan(module: AbstractModule, x1: Array, x2: Array):
		return jlx.cond(
			jnp.any(jnp.isnan(x1) | jnp.isnan(x2)),
			lambda _: jnp.asarray(jnp.nan),
			lambda _: module.pairwise(x1, x2),
			None
		)

	@staticmethod
	@filter_jit
	def cross_cov_vector(module: AbstractModule, x1: Array, x2: Array) -> Array:
		return vmap(NaNDenseEngine.pairwise_if_not_nan, in_axes=(None, None, 0))(module, x1, x2)

	@staticmethod
	@filter_jit
	def cross_cov_vector_if_not_nan(module: AbstractModule, x1: Array, x2: Array) -> Array:
		return jlx.cond(
			jnp.any(jnp.isnan(x1)),
			lambda _: jnp.full(len(x2), jnp.nan),
			lambda _: NaNDenseEngine.cross_cov_vector(module, x1, x2),
			None
		)

	@staticmethod
	@filter_jit
	def cross_cov_matrix(module: AbstractModule, x1: Array, x2: Array) -> Array:
		return vmap(NaNDenseEngine.cross_cov_vector_if_not_nan, in_axes=(None, 0, None))(module, x1, x2)


class NoJitDenseEngine(AbstractEngine):
	""" Engine optimised for efficiency in an environment where jax's jit is not available. """
	@staticmethod
	def __call__(module: AbstractModule, x1: Array, x2: Array, *args, **kwargs) -> Array:
		# Turn scalar inputs into vectors
		x1, x2 = jnp.atleast_1d(x1), jnp.atleast_1d(x2)

		# Call the appropriate method
		if jnp.ndim(x1) == 1 and jnp.ndim(x2) == 1:
			return module.pairwise(x1, x2)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 1:
			return vmap(module.pairwise, in_axes=(0, None))(x1, x2)
		elif jnp.ndim(x1) == 1 and jnp.ndim(x2) == 2:
			return vmap(module.pairwise, in_axes=(None, 0))(x1, x2)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 2:
			return vmap(vmap(module.pairwise, in_axes=(None, 0)), in_axes=(0, None))(x1, x2)
		else:
			raise ValueError(
				f"Invalid input dimensions: x1 has shape {x1.shape}, x2 has shape {x2.shape}. "
				"Expected scalar, 1D or 2D arrays as inputs."
			)
