import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel
from ..distances import squared_euclidean_distance
from .StationaryKernel import StaticStationaryKernel


class StaticRationalQuadraticKernel(StaticStationaryKernel):
	distance_func = squared_euclidean_distance

	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the Rational Quadratic kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (length_scale, alpha)
		:param x1: scalar array
		:param x2: scalar array
		:return: covariance value (scalar)
		"""
		kern = eqx.combine(kern)
		squared_dist = cls.distance_func(x1, x2)

		base = 1 + squared_dist / (2 * kern.alpha * kern.length_scale**2)  # type: ignore[attr-defined]

		return jnp.power(base, -kern.alpha)  # type: ignore[no-any-return,attr-defined]


class RationalQuadraticKernel(AbstractKernel):
	"""
	Rational Quadratic Kernel

	All parameters (length_scale, alpha) are always positive.
	Internally, they may be stored in unconstrained space.
	"""

	_unconstrained_length_scale: Array = eqx.field(converter=jnp.asarray)
	_unconstrained_alpha: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticRationalQuadraticKernel

	def __init__(self, length_scale, alpha):
		"""
		Initialize the Rational Quadratic kernel.

		Args:
			length_scale: length scale parameter (ℓ, must be positive)
			alpha: relative weighting of large-scale and small-scale variations (α, must be positive)

		Raises:
			ValueError: If any parameter is not positive
		"""
		# Validate positivity
		length_scale = jnp.array(length_scale)
		alpha = jnp.array(alpha)

		length_scale = eqx.error_if(
			length_scale, jnp.any(length_scale <= 0), "length_scale must be positive."
		)
		alpha = eqx.error_if(alpha, jnp.any(alpha <= 0), "alpha must be positive.")

		# Initialize parent (locks config)
		super().__init__()

		# Transform to unconstrained space
		from ..transforms import to_unconstrained

		self._unconstrained_length_scale = to_unconstrained(jnp.asarray(length_scale))
		self._unconstrained_alpha = to_unconstrained(jnp.asarray(alpha))

	@property
	def length_scale(self) -> Array:
		"""Get the length scale in constrained space (always positive)."""
		from ..transforms import to_constrained

		return to_constrained(self._unconstrained_length_scale)

	@property
	def alpha(self) -> Array:
		"""Get the alpha parameter in constrained space (always positive)."""
		from ..transforms import to_constrained

		return to_constrained(self._unconstrained_alpha)
