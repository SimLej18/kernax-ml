import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel, StaticAbstractKernel


# Matern 3/2 Kernel defined in Rasmussen and Williams (2006), section 4.2
class StaticMatern32Kernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: Array, x2: Array) -> Array:
		"""
		Compute the Matern 3/2 kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (length_scale)
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		r = jnp.linalg.norm(x1 - x2)  # Euclidean distance
		sqrt3_r_div_l = (jnp.sqrt(3) * r) / kern.length_scale
		return (1.0 + sqrt3_r_div_l) * jnp.exp(-sqrt3_r_div_l)  # type: ignore[no-any-return]


class Matern32Kernel(AbstractKernel):
	length_scale: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticMatern32Kernel

	def __init__(self, length_scale):
		super().__init__()
		self.length_scale = length_scale
