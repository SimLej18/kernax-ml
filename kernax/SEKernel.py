from functools import partial

from jax import jit, Array
from jax import numpy as jnp
import equinox as eqx

from kernax import StaticAbstractKernel, AbstractKernel


class StaticSEKernel(StaticAbstractKernel):
	@classmethod
	@partial(jit, static_argnums=(0,))
	def pairwise_cov(cls, kern: AbstractKernel, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: the kernel to use, containing a `length_scale` parameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return jnp.exp(-0.5 * ((x1 - x2) @ (x1 - x2)) / kern.length_scale ** 2)


class SEKernel(AbstractKernel):
	"""
	Squared Exponential (aka "RBF" or "Gaussian") Kernel
	"""
	length_scale: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticSEKernel

	def __init__(self, length_scale):
		super().__init__()
		self.length_scale = length_scale
