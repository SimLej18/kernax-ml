import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel
from .DotProductKernel import StaticDotProductKernel


class StaticPolynomialKernel(StaticDotProductKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the polynomial kernel covariance value between two vectors.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		kern = eqx.combine(kern)
		dp = cls.distance_func(x1, x2)
		return jnp.pow(kern.gamma * dp + kern.constant, kern.degree)  # type: ignore[attr-defined]


class PolynomialKernel(AbstractKernel):
	"""
	Polynomial Kernel

	Parameter gamma is always positive.
	Parameter constant can be any real value.
	Parameter degree is a static integer.
	"""

	degree: int = eqx.field(static=True)
	_unconstrained_gamma: Array = eqx.field(converter=jnp.asarray)
	constant: Array = eqx.field(converter=jnp.asarray)

	static_class = StaticPolynomialKernel

	def __init__(self, degree: int, gamma: float = 1.0, constant: float = 0.0):
		"""
		Initialize the Polynomial kernel.

		Args:
			degree: Degree of the polynomial (must be positive integer)
			gamma: Scale factor (must be positive)
			constant: Independent term (can be any real value)

		Raises:
			ValueError: If degree is not positive or gamma is not positive
		"""
		# Validate degree
		if degree <= 0:
			raise ValueError(f"degree must be a positive integer, got {degree}")

		# Validate gamma positivity
		gamma_array = jnp.array(gamma)
		gamma_array = eqx.error_if(gamma_array, jnp.any(gamma_array <= 0), "gamma must be positive.")

		# Initialize parent (locks config)
		super().__init__()

		# Set static degree
		self.degree = degree

		# Transform gamma to unconstrained space
		from ..transforms import to_unconstrained

		self._unconstrained_gamma = to_unconstrained(jnp.asarray(gamma_array))

		# constant can be any value, no transformation needed
		self.constant = jnp.asarray(constant)

	@property
	def gamma(self) -> Array:
		"""Get the gamma parameter in constrained space (always positive)."""
		from ..transforms import to_constrained

		return to_constrained(self._unconstrained_gamma)
