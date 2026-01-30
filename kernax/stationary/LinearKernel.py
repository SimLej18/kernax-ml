import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel, StaticAbstractKernel


class StaticLinearKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: Array, x2: Array) -> Array:
		"""
		Compute the linear kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (variance, c).
		:param x1: scalar array.
		:param x2: scalar array.
		:return: scalar array (covariance value).
		"""
		x1_shifted = x1 - kern.offset_c
		x2_shifted = x2 - kern.offset_c

		# Compute the dot product of the shifted vectors
		dot_product = jnp.sum(x1_shifted * x2_shifted)

		return kern.variance_b + kern.variance_v * dot_product  # type: ignore[no-any-return]


class LinearKernel(AbstractKernel):
	variance_b: Array = eqx.field(converter=jnp.asarray)
	variance_v: Array = eqx.field(converter=jnp.asarray)
	offset_c: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticLinearKernel

	def __init__(self, variance_b, variance_v, offset_c):
		"""
		:param variance_b: Bias variance (σ²_b). Controls the vertical offset.
		:param variance_v: Weight variance (σ²_v). Controls the slope.
		:param offset_c: Input offset (c). Determines the crossing point of the functions.
		"""
		super().__init__()
		self.variance_b = variance_b
		self.variance_v = variance_v
		self.offset_c = offset_c
