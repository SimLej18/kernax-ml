from __future__ import annotations
from typing import Callable
import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp
from .StationaryKernel import AbstractStationaryKernel
from ..distances import squared_euclidean_distance
from ..engines import AbstractEngine, DenseEngine
from ..parametrisations import AbstractParametrisation, LogExpParametrisation


class SEKernel(AbstractStationaryKernel):
	"""
	Squared Exponential (aka "RBF" or "Gaussian") Kernel

	The length_scale parameter is constrained to stay positive and provide smoother optimisation.
	"""
	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	_length_scale_parametrisation: AbstractParametrisation = eqx.field()
	_length_scale: Array = eqx.field(converter=jnp.asarray)

	@property
	def length_scale(self):
		return self._length_scale_parametrisation.unwrap(self._length_scale)

	def __init__(self,
	             length_scale: float | Array,
	             length_scale_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             distance_function: Callable = squared_euclidean_distance,
	             engine: AbstractEngine = DenseEngine):
		# Assert length_scale is positive
		length_scale = jnp.asarray(length_scale)
		if jnp.any(length_scale <= 0):
			raise ValueError("`length_scale` must be positive.")

		self.distance_function = distance_function
		self._length_scale_parametrisation = length_scale_parametrisation
		self._length_scale = self._length_scale_parametrisation.wrap(length_scale)
		self.engine = engine

	@filter_jit
	def pairwise(self, x1: Array, x2: Array):
		return jnp.exp(-0.5 * self.distance_function(x1, x2) / self.length_scale**2)

	def replace(self, length_scale: None|float|Array = None, **kwargs) -> SEKernel:
		if length_scale is None:
			return self  # No change to make

		# Assert length_scale is positive
		length_scale = jnp.asarray(length_scale)
		if jnp.any(length_scale <= 0):
			raise ValueError("`length_scale` must be positive.")

		return eqx.tree_at(
			lambda k: k._length_scale,
			self,
			jnp.broadcast_to(
				self._length_scale_parametrisation.wrap(jnp.asarray(length_scale)),
				self._length_scale.shape)
		)

RBFKernel = SEKernel  # Equivalent
