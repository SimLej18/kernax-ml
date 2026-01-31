import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel
from .DotProductKernel import StaticDotProductKernel


class StaticSigmoidKernel(StaticDotProductKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the sigmoid kernel covariance value between two vectors.

		Formula: tanh(α⟨x, x'⟩ + c)

		:param kern: kernel instance containing the hyperparameters (alpha, constant)
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		kern = eqx.combine(kern)
		dp = cls.distance_func(x1, x2)
		return jnp.tanh(kern.alpha * dp + kern.constant)  # type: ignore[attr-defined]


class SigmoidKernel(AbstractKernel):
	"""
	Sigmoid (Hyperbolic Tangent) Kernel

	Formula: tanh(α⟨x, x'⟩ + c)

	Parameter alpha must be positive.
	Parameter constant can be any real value.
	"""

	_unconstrained_alpha: Array = eqx.field(converter=jnp.asarray)
	constant: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticSigmoidKernel

	def __init__(self, alpha: float = 1.0, constant: float = 0.0):
		"""
		Initialize the Sigmoid kernel.

		Args:
			alpha: Scale factor (must be positive). Controls the steepness.
			constant: Independent term (can be any real value). Shifts the activation.

		Raises:
			ValueError: If alpha is not positive
		"""
		# Validate alpha positivity
		alpha_array = jnp.array(alpha)
		alpha_array = eqx.error_if(alpha_array, jnp.any(alpha_array <= 0), "alpha must be positive.")

		# Initialize parent (locks config)
		super().__init__()

		# Transform alpha to unconstrained space
		from ..transforms import to_unconstrained

		self._unconstrained_alpha = to_unconstrained(jnp.asarray(alpha_array))

		# constant can be any value, no transformation needed
		self.constant = jnp.asarray(constant)

	@property
	def alpha(self) -> Array:
		"""Get the alpha parameter in constrained space (always positive)."""
		from ..transforms import to_constrained

		return to_constrained(self._unconstrained_alpha)
