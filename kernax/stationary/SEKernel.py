import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel, StaticAbstractKernel


class StaticSEKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		kern = eqx.combine(kern)
		return jnp.exp(-0.5 * (jnp.sum((x1 - x2) ** 2)) / kern.length_scale**2)  # type: ignore[attr-defined]


class SEKernel(AbstractKernel):
	"""
	Squared Exponential (aka "RBF" or "Gaussian") Kernel
	"""

	length_scale: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticSEKernel

	def __init__(self, length_scale):
		super().__init__()
		self.length_scale = length_scale


class RBFKernel(SEKernel):
	"""
	Same as SEKernel
	"""

	pass
