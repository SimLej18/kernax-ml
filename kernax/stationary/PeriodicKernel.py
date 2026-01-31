import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel
from ..distances import euclidean_distance
from .StationaryKernel import StaticStationaryKernel


class StaticPeriodicKernel(StaticStationaryKernel):
	distance_func = euclidean_distance

	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the periodic kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (length_scale, variance, period).
		:param x1: scalar array
		:param x2: scalar array
		:return: covariance value (scalar)
		"""
		kern = eqx.combine(kern)
		dist = cls.distance_func(x1, x2)

		return kern.variance * jnp.exp(  # type: ignore[no-any-return,attr-defined]
			-2 * jnp.sin(jnp.pi * dist / kern.period) ** 2 / kern.length_scale**2  # type: ignore[attr-defined]
		)


class PeriodicKernel(AbstractKernel):
	"""
	Periodic Kernel

	All parameters (length_scale, variance, period) are always positive.
	Internally, they may be stored in unconstrained space.
	"""

	_unconstrained_length_scale: Array = eqx.field(converter=jnp.asarray)
	_unconstrained_variance: Array = eqx.field(converter=jnp.asarray)
	_unconstrained_period: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticPeriodicKernel

	def __init__(self, length_scale, variance, period):
		"""
		Initialize the Periodic kernel.

		Args:
			length_scale: length scale parameter (ℓ, must be positive)
			variance: variance parameter (σ², must be positive)
			period: period parameter (p, must be positive)

		Raises:
			ValueError: If any parameter is not positive
		"""
		# Validate positivity
		length_scale = jnp.array(length_scale)
		variance = jnp.array(variance)
		period = jnp.array(period)

		length_scale = eqx.error_if(
			length_scale, jnp.any(length_scale <= 0), "length_scale must be positive."
		)
		variance = eqx.error_if(variance, jnp.any(variance <= 0), "variance must be positive.")
		period = eqx.error_if(period, jnp.any(period <= 0), "period must be positive.")

		# Initialize parent (locks config)
		super().__init__()

		# Transform to unconstrained space
		from ..transforms import to_unconstrained

		self._unconstrained_length_scale = to_unconstrained(jnp.asarray(length_scale))
		self._unconstrained_variance = to_unconstrained(jnp.asarray(variance))
		self._unconstrained_period = to_unconstrained(jnp.asarray(period))

	@property
	def length_scale(self) -> Array:
		"""Get the length scale in constrained space (always positive)."""
		from ..transforms import to_constrained

		return to_constrained(self._unconstrained_length_scale)

	@property
	def variance(self) -> Array:
		"""Get the variance in constrained space (always positive)."""
		from ..transforms import to_constrained

		return to_constrained(self._unconstrained_variance)

	@property
	def period(self) -> Array:
		"""Get the period in constrained space (always positive)."""
		from ..transforms import to_constrained

		return to_constrained(self._unconstrained_period)
