import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel, StaticAbstractKernel


class StaticRationalQuadraticKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: Array, x2: Array) -> Array:
		"""
		Compute the Rational Quadratic kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (variance, length_scale, alpha)
		:param x1: scalar array
		:param x2: scalar array
		:return: covariance value (scalar)
		"""
		squared_dist = jnp.sum((x1 - x2) ** 2)

		base = 1 + squared_dist / (2 * kern.alpha * kern.length_scale**2)

		return jnp.power(base, -kern.alpha)


class RationalQuadraticKernel(AbstractKernel):
	length_scale: Array = eqx.field(converter=jnp.asarray)
	alpha: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticRationalQuadraticKernel

	def __init__(self, length_scale, alpha):
		"""
		:param length_scale: length scale parameter (ℓ)
		:param variance: variance (σ²)
		:param alpha: relative weighting of large-scale and small-scale variations (α)
		"""
		super().__init__()
		self.length_scale = length_scale
		self.alpha = alpha
