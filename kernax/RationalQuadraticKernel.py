from functools import partial

from jax import jit, Array
from jax import numpy as jnp
import equinox as eqx

from kernax import StaticAbstractKernel, AbstractKernel


class StaticRationalQuadraticKernel(StaticAbstractKernel):
	@classmethod
	@partial(jit, static_argnums=(0,))
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the Rational Quadratic kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (variance, length_scale, alpha)
		:param x1: scalar array
		:param x2: scalar array
		:return: covariance value (scalar)
		"""
		squared_dist = jnp.sum((x1 - x2) ** 2)
		
		base = 1 + squared_dist / (2 * kern.alpha * kern.length_scale ** 2)
		
		return kern.variance * jnp.power(base, -kern.alpha)


class RationalQuadraticKernel(AbstractKernel):
	length_scale: Array = eqx.field(converter=jnp.asarray)
	variance: Array = eqx.field(converter=jnp.asarray)
	alpha: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticRationalQuadraticKernel

	def __init__(self, length_scale, variance, alpha):
		"""
		:param length_scale: length scale parameter (ℓ)
		:param variance: variance (σ²)
		:param alpha: relative weighting of large-scale and small-scale variations (α)
		"""
		super().__init__()
		self.length_scale = length_scale
		self.variance = variance
		self.alpha = alpha
