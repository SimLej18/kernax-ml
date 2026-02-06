import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..engines import SafeDiagonalEngine
from .ConstantKernel import AbstractKernel, StaticConstantKernel
from ..transforms import to_constrained, to_unconstrained


class WhiteNoiseKernel(AbstractKernel):
	"""
	White noise kernel that returns a constant value only on the diagonal.

	This kernel is equivalent to a ConstantKernel with a SafeDiagonalEngine computation engine.
	It returns the noise value when inputs are equal, and 0 otherwise, resulting in a diagonal
	covariance matrix.
	"""

	_raw_noise: Array = eqx.field(converter=jnp.asarray)

	@property
	def noise(self) -> Array:
		return to_constrained(self._raw_noise)

	@property
	def value(self):  # As we use the StaticConstantKernel pairwise_cov, we need to provide a 'value' property that returns the noise level
		return to_constrained(self._raw_noise)

	static_class = StaticConstantKernel

	def __init__(self, noise=1.0, **kwargs):
		# Assert noise is positive
		noise = jnp.array(noise)
		noise = eqx.error_if(noise, jnp.any(noise < 0), "noise must be positive or nul.")

		# Set the computation engine to SafeDiagonalEngine for diagonal computation
		kwargs['computation_engine'] = SafeDiagonalEngine
		super().__init__(**kwargs)

		# Transform to unconstrained space
		self._raw_noise = to_unconstrained(jnp.asarray(noise))

