from __future__ import annotations
from typing import Callable
import equinox as eqx
from jax import Array
from jax import numpy as jnp
from equinox import filter_jit
from .StationaryKernel import AbstractStationaryKernel
from ..engines import AbstractEngine, DenseEngine
from ..distances import equality
from ..parametrisations import AbstractParametrisation, LogExpParametrisation


class WhiteNoiseKernel(AbstractStationaryKernel):
	"""
	White noise kernel that returns a constant value only on the diagonal, aka where x1==x2.
	"""
	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	_noise_parametrisation: AbstractParametrisation = eqx.field()
	_noise: Array = eqx.field(converter=jnp.asarray)

	@property
	def noise(self) -> Array:
		return self._noise_parametrisation.unwrap(self._noise)

	def __init__(self,
	             noise: float | Array,
	             noise_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             engine: AbstractEngine = DenseEngine):
		# Assert noise is positive
		noise = jnp.asarray(noise)

		if jnp.any(noise < 0):
			raise ValueError("`noise` must be positive or nul.")

		self.distance_function = equality
		self._noise_parametrisation = noise_parametrisation
		self._noise = self._noise_parametrisation.wrap(noise)
		self.engine = engine

	@filter_jit
	def pairwise(self, x1: Array, x2: Array):
		return self.distance_function(x1, x2)

	def replace(self, noise: None|float|Array = None, **kwargs) -> WhiteNoiseKernel:
		if noise is None:
			return self  # No change to make

		noise = jnp.asarray(noise)

		if jnp.any(noise < 0):
			raise ValueError("`noise` must be positive or nul.")

		return eqx.tree_at(
			lambda k: k._noise,
			self,
			jnp.broadcast_to(
				self._noise_parametrisation.wrap(noise),
				self._noise.shape)
		)
