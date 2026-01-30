import equinox as eqx
import jax.numpy as jnp
from equinox import filter_jit
from jax import Array

from ..AbstractKernel import AbstractKernel, StaticAbstractKernel
from ..utils import format_jax_array


class StaticConstantKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return kern.value  # type: ignore[no-any-return]  # The constant value is returned regardless of the inputs


class ConstantKernel(AbstractKernel):
	value: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticConstantKernel

	def __init__(self, value=1.0):
		"""
		Instantiates a constant kernel with the given value.

		:param value: the value of the constant kernel
		"""
		super().__init__()
		self.value = value

	def __str__(self):
		return format_jax_array(self.value)
