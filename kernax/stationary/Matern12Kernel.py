import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel
from ..distances import euclidean_distance
from .StationaryKernel import StaticStationaryKernel


# Matern 1/2 (Exponential) Kernel defined in Rasmussen and Williams (2006), section 4.2
class StaticMatern12Kernel(StaticStationaryKernel):
	distance_func = euclidean_distance

	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the Matern 1/2 kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (length_scale)
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		kern = eqx.combine(kern)
		r = cls.distance_func(x1, x2)
		return jnp.exp(-r / kern.length_scale)  # type: ignore[attr-defined]


class Matern12Kernel(AbstractKernel):
	"""
	Matern 1/2 (aka Exponential) Kernel

	The length_scale parameter is always positive. Internally, it may be stored in an
	unconstrained space depending on the global configuration.
	"""

	_unconstrained_length_scale: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticMatern12Kernel

	def __init__(self, length_scale):
		"""
		Initialize the Matern 1/2 kernel with a length scale parameter.

		Args:
			length_scale: Length scale parameter (must be positive)

		Raises:
			ValueError: If length_scale is not positive
		"""
		# Validate positivity
		length_scale = jnp.array(length_scale)
		length_scale = eqx.error_if(
			length_scale, jnp.any(length_scale <= 0), "length_scale must be positive."
		)

		# Initialize parent (locks config)
		super().__init__()

		# Transform to unconstrained space
		from ..transforms import to_unconstrained

		self._unconstrained_length_scale = to_unconstrained(jnp.asarray(length_scale))

	@property
	def length_scale(self) -> Array:
		"""Get the length scale in constrained space (always positive)."""
		from ..transforms import to_constrained

		return to_constrained(self._unconstrained_length_scale)
