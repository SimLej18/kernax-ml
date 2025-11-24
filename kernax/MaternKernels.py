import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from kernax import AbstractKernel, StaticAbstractKernel


# Matern 1/2 (Exponential) Kernel defined in Rasmussen and Williams (2006), section 4.2
class StaticMatern12Kernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the Matern 1/2 kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (length_scale)
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		r = jnp.linalg.norm(x1 - x2)  # Euclidean distance
		return jnp.exp(-r / kern.length_scale)


class Matern12Kernel(AbstractKernel):
	length_scale: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticMatern12Kernel

	def __init__(self, length_scale):
		super().__init__()
		self.length_scale = length_scale


# Matern 3/2 Kernel defined in Rasmussen and Williams (2006), section 4.2
class StaticMatern32Kernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the Matern 3/2 kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (length_scale)
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		r = jnp.linalg.norm(x1 - x2)  # Euclidean distance
		sqrt3_r_div_l = (jnp.sqrt(3) * r) / kern.length_scale
		return (1.0 + sqrt3_r_div_l) * jnp.exp(-sqrt3_r_div_l)


class Matern32Kernel(AbstractKernel):
	length_scale: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticMatern32Kernel

	def __init__(self, length_scale):
		super().__init__()
		self.length_scale = length_scale


# Matern 5/2 Kernel defined in Rasmussen and Williams (2006), section 4.2
class StaticMatern52Kernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the Matern 5/2 kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (length_scale)
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		r = jnp.linalg.norm(x1 - x2)  # Euclidean distance
		sqrt5_r_div_l = (jnp.sqrt(5) * r) / kern.length_scale
		return (1.0 + sqrt5_r_div_l + (5.0 / 3.0) * (r / kern.length_scale) ** 2) * jnp.exp(
			-sqrt5_r_div_l
		)


class Matern52Kernel(AbstractKernel):
	length_scale: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticMatern52Kernel

	def __init__(self, length_scale):
		super().__init__()
		self.length_scale = length_scale
