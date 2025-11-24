from functools import partial

from jax import jit, Array
from jax import numpy as jnp
import equinox as eqx

from kernax import StaticAbstractKernel, AbstractKernel


class StaticPeriodicKernel(StaticAbstractKernel):
	@classmethod
	@partial(jit, static_argnums=(0,))
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
        Compute the periodic kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (length_scale, variance, period).
		:param x1: scalar array
		:param x2: scalar array
		:return: covariance value (scalar)
		"""
		dist = jnp.linalg.norm(x1 - x2)

		return kern.variance * jnp.exp(-2 * jnp.sin(jnp.pi * dist / kern.period)**2 / kern.length_scale**2)


class PeriodicKernel(AbstractKernel):
	length_scale: Array = eqx.field(converter=jnp.asarray)
	variance: Array = eqx.field(converter=jnp.asarray)
	period: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticPeriodicKernel

	def __init__(self, length_scale, variance, period):
		"""
		:param length_scale: length scale parameter (ℓ)
		:param variance: variance parameter (σ²)
		:param period: period parameter (p)
		"""
		super().__init__()
		self.length_scale = length_scale
		self.variance = variance
		self.period = period