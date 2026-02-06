import equinox as eqx
from jax import Array
from jax import numpy as jnp

from .ConstantKernel import AbstractKernel, StaticConstantKernel
from ..transforms import to_constrained, to_unconstrained


class VarianceKernel(AbstractKernel):
	"""
	Variance kernel that returns a constant value everywhere. Used to multiply with other kernels.

	This kernel is functionally equivalent to a ConstantKernel but the implementation differs in
	two ways:
	1) The "value" parameter is renamed "variance" parameter, allowing easier modification when
	combining with another ConstantKernel (e.g. WhiteNoiseKernel)
	2) It forces a positivity constraint on the variance parameter

	"""

	_raw_variance: Array = eqx.field(converter=jnp.asarray)

	@property
	def variance(self) -> Array:
		return to_constrained(self._raw_variance)

	@property
	def value(self):  # As we use the StaticConstantKernel pairwise_cov, we need to provide a 'value' property that returns the variance
		return to_constrained(self._raw_variance)

	static_class = StaticConstantKernel

	def __init__(self, variance=1.0, **kwargs):
		# Assert noise is positive
		variance = jnp.array(variance)
		variance = eqx.error_if(variance, jnp.any(variance <= 0), "variance must be positive.")

		super().__init__(**kwargs)

		# Transform to unconstrained space
		self._raw_variance = to_unconstrained(jnp.asarray(variance))

