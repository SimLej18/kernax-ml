from functools import partial

import jax.numpy as jnp
from jax import jit, Array
import equinox as eqx

from kernax import StaticAbstractKernel, AbstractKernel


class StaticConstantKernel(StaticAbstractKernel):
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
		return kern.value  # The constant value is returned regardless of the inputs


class ConstantKernel(AbstractKernel):
	value: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticConstantKernel

	def __init__(self, value=1.):
		"""
		Instantiates a constant kernel with the given value.

		:param value: the value of the constant kernel
		"""
		super().__init__()
		self.value = value
