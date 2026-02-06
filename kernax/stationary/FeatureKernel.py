from jax import numpy as jnp
from jax import Array
import equinox as eqx
from equinox import filter_jit

from ..AbstractKernel import AbstractKernel
from ..distances import squared_euclidean_distance
from ..transforms import to_constrained, to_unconstrained
from .StationaryKernel import StaticStationaryKernel


class StaticFeatureKernel(StaticStationaryKernel):
	""" Static class allowing FeatureKernel to be used inside a BlockKernel. """

	distance_func = squared_euclidean_distance

	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		kern = eqx.combine(kern)

		# If length_scale or variance is a scalar, convert it to 2-element array for consistency
		if kern.length_scale.ndim == 0:
			kern_length_scale = jnp.array([kern.length_scale, kern.length_scale])
		else:
			kern_length_scale = kern.length_scale
		if kern.variance.ndim == 0:
			kern_variance = jnp.array([kern.variance, kern.variance])
		else:
			kern_variance = kern.variance

		# As the formula only involves diagonal matrices, we can compute directly with vectors
		sigma_diag = kern_length_scale[0] + kern_length_scale[1] + kern.length_scale_u  # Σ
		sigma_det = jnp.prod(sigma_diag)  # |Σ|

		# Compute the quadratic form: (x - x')^T Sigma^{-1} (x - x')
		# Since Sigma^{-1} is diagonal, this simplifies to sum of (diff_i^2 * sigma_inv_diag_i)
		quadratic_form = cls.distance_func(x1, x2) / sigma_diag

		return kern_variance[0] * kern_variance[1] / (((2 * jnp.pi)**(len(x1)/2)) * jnp.sqrt(sigma_det)) * jnp.exp(-0.5 * quadratic_form)


class FeatureKernel(AbstractKernel):
	"""
	Feature Kernel with multiple positive-constrained length scales and variances.

	All parameters (length_scale_1, length_scale_2, length_scale_u, variance_1, variance_2)
	are constrained to be positive.
	"""

	_raw_length_scale: Array = eqx.field(converter=jnp.asarray)
	_raw_length_scale_u: Array = eqx.field(converter=jnp.asarray)
	_raw_variance: Array = eqx.field(converter=jnp.asarray)

	static_class = StaticFeatureKernel

	def __init__(self, length_scale, length_scale_u, variance, **kwargs):
		"""
		Initialize the Feature kernel.

		:param length_scale: length scale parameter - must be positive
		:param length_scale_u: uncertainty length scale parameter - must be positive
		:param variance: variance parameter - must be positive
		"""
		# Validate all parameters are positive
		length_scale = jnp.array(length_scale)
		length_scale_u = jnp.array(length_scale_u)
		variance = jnp.array(variance)

		length_scale = eqx.error_if(length_scale, jnp.any(length_scale <= 0), "length_scale must be positive.")
		length_scale_u = eqx.error_if(length_scale_u, jnp.any(length_scale_u <= 0), "length_scale_u must be positive.")
		variance = eqx.error_if(variance, jnp.any(variance <= 0), "variance must be positive.")

		# Initialize parent
		super().__init__(**kwargs)

		# Store in unconstrained space
		self._raw_length_scale = to_unconstrained(jnp.asarray(length_scale))
		self._raw_length_scale_u = to_unconstrained(jnp.asarray(length_scale_u))
		self._raw_variance = to_unconstrained(jnp.asarray(variance))

	@property
	def length_scale(self) -> Array:
		"""Get length_scale_1 in constrained (positive) space."""
		return to_constrained(self._raw_length_scale)

	@property
	def length_scale_u(self) -> Array:
		"""Get length_scale_u in constrained (positive) space."""
		return to_constrained(self._raw_length_scale_u)

	@property
	def variance(self) -> Array:
		"""Get variance_1 in constrained (positive) space."""
		return to_constrained(self._raw_variance)
