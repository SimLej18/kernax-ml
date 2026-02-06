import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel
from ..distances import squared_euclidean_distance
from ..transforms import to_constrained, to_unconstrained
from .StationaryKernel import StaticStationaryKernel


class StaticSEKernel(StaticStationaryKernel):
	distance_func = squared_euclidean_distance

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
		return jnp.exp(-0.5 * cls.distance_func(x1, x2) / kern.length_scale**2)  # type: ignore[attr-defined]


class SEKernel(AbstractKernel):
	"""
	Squared Exponential (aka "RBF" or "Gaussian") Kernel

	The length_scale parameter is always positive. Internally, it may be stored in an
	unconstrained space (log-space or softplus-inverse space) depending on the global
	configuration, which improves optimization stability.
	"""

	_raw_length_scale: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticSEKernel

	def __init__(self, length_scale, **kwargs):
		"""
		Initialize the SE kernel with a length scale parameter.

		Args:
			length_scale: Length scale parameter (must be positive). This is provided
				in the constrained space (positive values) and will be converted to
				the appropriate unconstrained space based on config.parameter_transform.
			**kwargs: Additional arguments passed to AbstractKernel (e.g., computation_engine)

		Raises:
			ValueError: If length_scale is not positive
		"""
		# Assert length_scale is positive
		length_scale = jnp.array(length_scale)
		length_scale = eqx.error_if(length_scale, jnp.any(length_scale <= 0), "length_scale must be positive.")

		# Initialize parent (marks kernels as instantiated, locks config)
		super().__init__(**kwargs)

		# Transform to unconstrained space
		self._raw_length_scale = to_unconstrained(jnp.asarray(length_scale))

	@property
	def length_scale(self) -> Array:
		"""
		Get the length scale in constrained space (always positive).

		This property applies the forward transformation to convert from the internal
		unconstrained representation back to the constrained (positive) space.

		Returns:
			Length scale parameter (positive)
		"""
		return to_constrained(self._raw_length_scale)


class RBFKernel(SEKernel):
	"""
	Same as SEKernel
	"""

	pass
