import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel, StaticAbstractKernel


class StaticPolynomialKernel(StaticAbstractKernel):
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
		return jnp.pow(kern.gamma * (x1.T @ x2) + kern.constant, kern.degree)  # type: ignore[attr-defined]


class PolynomialKernel(AbstractKernel):
	"""
	Polynomial kernel.
	"""

	degree: int = eqx.field(static=True)
	gamma: Array = eqx.field(converter=jnp.asarray)
	constant: Array = eqx.field(converter=jnp.asarray)

	static_class = StaticPolynomialKernel

	def __init__(self, degree: int, gamma: float = 1.0, constant: float = 0.0):
		"""
		:param degree: degree of the polynomial
		:param gamma: scale factor
		:param constant: independent term
		"""
		super().__init__()
		self.degree = degree
		self.gamma = gamma  # type: ignore[assignment]  # Converter handles float to Array
		self.constant = constant  # type: ignore[assignment]  # Converter handles float to Array
