import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel
from ..distances import dot_product
from .DotProductKernel import StaticDotProductKernel


class StaticLinearKernel(StaticDotProductKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the linear kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (variance_b, variance_v, offset_c).
		:param x1: scalar array.
		:param x2: scalar array.
		:return: scalar array (covariance value).
		"""
		kern = eqx.combine(kern)
		x1_shifted = x1 - kern.offset_c  # type: ignore[attr-defined]
		x2_shifted = x2 - kern.offset_c  # type: ignore[attr-defined]

		# Compute the dot product of the shifted vectors
		dp = cls.distance_func(x1_shifted, x2_shifted)

		return kern.variance_b + kern.variance_v * dp  # type: ignore[no-any-return,attr-defined]


class LinearKernel(AbstractKernel):
	"""
	Linear Kernel

	Parameters variance_b and variance_v must be non-negative (>= 0).
	Parameter offset_c can be any real value.

	Note: Variances are NOT transformed (no log-exp trick) because they can be zero,
	which is incompatible with log-based transformations.
	"""

	variance_b: Array = eqx.field(converter=jnp.asarray)
	variance_v: Array = eqx.field(converter=jnp.asarray)
	offset_c: Array = eqx.field(converter=jnp.asarray)
	static_class = StaticLinearKernel

	def __init__(self, variance_b, variance_v, offset_c):
		"""
		Initialize the Linear kernel.

		Args:
			variance_b: Bias variance (σ²_b, must be non-negative). Controls the vertical offset.
			variance_v: Weight variance (σ²_v, must be non-negative). Controls the slope.
			offset_c: Input offset (c, can be any real value). Determines the crossing point.

		Raises:
			ValueError: If variance_b or variance_v is negative
		"""
		# Validate non-negativity for variances
		variance_b = jnp.array(variance_b)
		variance_v = jnp.array(variance_v)

		variance_b = eqx.error_if(
			variance_b, jnp.any(variance_b < 0), "variance_b must be non-negative."
		)
		variance_v = eqx.error_if(
			variance_v, jnp.any(variance_v < 0), "variance_v must be non-negative."
		)

		# Initialize parent (locks config)
		super().__init__()

		# Store parameters as-is (no transformation)
		# Variances can be 0, which is incompatible with log-based transforms
		self.variance_b = jnp.asarray(variance_b)
		self.variance_v = jnp.asarray(variance_v)
		self.offset_c = jnp.asarray(offset_c)
