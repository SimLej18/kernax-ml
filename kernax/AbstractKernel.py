from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import cond
import equinox as eqx


class StaticAbstractKernel:
	@classmethod
	@partial(jit, static_argnums=(0,))
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return jnp.array(jnp.nan)  # To be overwritten in subclasses

	@classmethod
	@partial(jit, static_argnums=(0,))
	def pairwise_cov_if_not_nan(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Returns NaN if either x1 or x2 is NaN, otherwise calls the compute_scalar method.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return cond(jnp.any(jnp.isnan(x1) | jnp.isnan(x2)),
		            lambda _: jnp.nan,
		            lambda _: cls.pairwise_cov(kern, x1, x2),
		            None)

	@classmethod
	@partial(jit, static_argnums=(0,))
	def cross_cov_vector(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel cross covariance values between an array of vectors (matrix) and a vector.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: vector array (N, )
		:param x2: scalar array
		:return: vector array (N, )
		"""
		return vmap(lambda x: cls.pairwise_cov_if_not_nan(kern, x, x2), in_axes=0)(x1)

	@classmethod
	@partial(jit, static_argnums=(0,))
	def cross_cov_vector_if_not_nan(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Returns an array of NaN if scalar is NaN, otherwise calls the compute_vector method.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return cond(jnp.any(jnp.isnan(x2)),
		            lambda _: jnp.full(len(x1), jnp.nan),
		            lambda _: cls.cross_cov_vector(kern, x1, x2),
		            None)

	@classmethod
	@partial(jit, static_argnums=(0,))
	def cross_cov_matrix(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance matrix between two vector arrays.

		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:return: matrix array (N, M)
		"""
		return vmap(lambda x: cls.cross_cov_vector_if_not_nan(kern, x2, x), in_axes=0)(x1)


class AbstractKernel(eqx.Module):
	"""
	# TODO: check Equinox __str__ and __repr__ methods and adapt if needed
	def __str__(self):
		return f"{self.__class__.__name__}({', '.join([f'{key}={value}' for key, value in self.__dict__.items() if key not in self.static_attributes])})"

	def __repr__(self):
		return str(self)
	"""

	@jit
	def __call__(self, x1, x2=None):
		# If no x2 is provided, we compute the covariance between x1 and itself
		if x2 is None:
			x2 = x1

		# Turn scalar inputs into vectors
		x1, x2 = jnp.atleast_1d(x1), jnp.atleast_1d(x2)

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
				"Expected 1D, 2D arrays or 3D arrays for batched inputs."
			)

	def __add__(self, other):
		from kernax.OperatorKernels import SumKernel
		return SumKernel(self, other)

	def __radd__(self, other):
		from kernax.OperatorKernels import SumKernel
		return SumKernel(other, self)

	def __neg__(self):
		from kernax.WrapperKernels import NegKernel
		return NegKernel(self)

	def __mul__(self, other):
		from kernax.OperatorKernels import ProductKernel
		return ProductKernel(self, other)

	def __rmul__(self, other):
		from kernax.OperatorKernels import ProductKernel
		return ProductKernel(other, self)
