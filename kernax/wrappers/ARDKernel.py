from typing import Iterable
import equinox as eqx
import jax.numpy as jnp
from equinox import filter_jit
from jax import Array
from ..stationary.StationaryKernel import AbstractStationaryKernel
from ..parametrisations import AbstractParametrisation, LogExpParametrisation
from .WrapperModule import AbstractWrapperModule


class ARDKernel(AbstractWrapperModule):
	"""
	Wrapper kernel to apply Automatic Relevance Determination (ARD) to the inputs before passing them to the inner kernel.
	Each input dimension is scaled by a separate length scale hyperparameter.
	"""
	inner: AbstractStationaryKernel
	_length_scales_parametrisation: AbstractParametrisation = eqx.field()
	_length_scales: Array = eqx.field(converter=jnp.asarray)

	@property
	def length_scales(self):
		return self._length_scales_parametrisation.unwrap(self._length_scales)


	def __init__(self,
	             inner: AbstractStationaryKernel,
	             length_scales: Iterable[float] | Array,
	             length_scales_parametrisation: AbstractParametrisation = LogExpParametrisation()):
		"""
		:param inner: the kernel to wrap, must be an instance of AbstractKernel. Every 'length_scale'
		attribute inside the inner kernel should have the value 1 and be parametrised with a
		NonTrainableParametrisation.
		:param length_scales: the length scales for each input dimension (1D array of floats or Array). Must be positive
		"""
		length_scales = jnp.asarray(length_scales)
		if jnp.any(length_scales <= 0):
			raise ValueError("input length scales must be positive")

		self.inner = inner
		self._length_scales_parametrisation = length_scales_parametrisation
		self._length_scales = self._length_scales_parametrisation.wrap(length_scales)

	@filter_jit
	def __call__(self, x1: Array, x2: None | Array = None) -> Array:
		if x1.shape[-1] != self.length_scales.shape[-1]:
			raise ValueError(f"input shape {x1.shape} does not match length scale shape {self.length_scales.shape}")

		if x2 is None:
			x2 = x1

		return self.inner(x1 / self.length_scales, x2 / self.length_scales)

	def replace(self,
	            length_scales: Iterable[float] | Array = None,
				**kwargs) -> Array:
		if length_scales is None:
			return self

		length_scales = jnp.asarray(length_scales)

		if jnp.any(length_scales <= 0):
			raise ValueError("input length scales must all be positive")

		return eqx.tree_at(
			lambda k: k._length_scales,
			self,
			self._length_scales_parametrisation.wrap(length_scales))
