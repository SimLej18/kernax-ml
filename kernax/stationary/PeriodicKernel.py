from __future__ import annotations
from typing import Callable
import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp
from .StationaryKernel import AbstractStationaryKernel
from ..distances import euclidean_distance
from ..engines import AbstractEngine, DenseEngine
from ..parametrisations import AbstractParametrisation, LogExpParametrisation


class PeriodicKernel(AbstractStationaryKernel):
	"""Periodic Kernel"""

	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	_length_scale_parametrisation: AbstractParametrisation = eqx.field()
	_period_parametrisation: AbstractParametrisation = eqx.field()
	_length_scale: Array = eqx.field(converter=jnp.asarray)
	_period: Array = eqx.field(converter=jnp.asarray)

	@property
	def length_scale(self) -> Array:
		return self._length_scale_parametrisation.unwrap(self._length_scale)


	@property
	def period(self) -> Array:
		return self._period_parametrisation.unwrap(self._period)

	def __init__(self,
	             length_scale: float | Array,
	             period: float | Array,
	             length_scale_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             period_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             distance_function: Callable = euclidean_distance,
	             engine: AbstractEngine = DenseEngine):
		length_scale = jnp.asarray(length_scale)
		period = jnp.asarray(period)

		if jnp.any(length_scale <= 0):
			raise ValueError("`length_scale` must be positive.")
		if jnp.any(period <= 0):
			raise ValueError("`period` must be positive.")

		self.distance_function = distance_function
		self._length_scale_parametrisation = length_scale_parametrisation
		self._period_parametrisation = period_parametrisation
		self._length_scale = self._length_scale_parametrisation.wrap(length_scale)
		self._period = self._period_parametrisation.wrap(period)
		self.engine = engine

	@filter_jit
	def pairwise(self, x1: Array, x2: Array) -> Array:
		dist = self.distance_function(x1, x2)
		return jnp.exp(-2 * jnp.sin(jnp.pi * dist / self.period)**2 / self.length_scale**2)

	def replace(self,
	            length_scale: None | float | Array = None,
	            period: None | float | Array = None,
	            **kwargs) -> PeriodicKernel:
		new_kernel = self

		if length_scale is not None:
			length_scale = jnp.asarray(length_scale)

			if jnp.any(length_scale <= 0):
				raise ValueError("`length_scale` must be positive.")

			new_kernel = eqx.tree_at(
				lambda k: k._length_scale,
				new_kernel,
				jnp.broadcast_to(
					self._length_scale_parametrisation.wrap(length_scale),
					self._length_scale.shape)
			)

		if period is not None:
			period = jnp.asarray(period)

			if jnp.any(period <= 0):
				raise ValueError("`period` must be positive.")

			new_kernel = eqx.tree_at(
				lambda k: k._period,
				new_kernel,
				jnp.broadcast_to(
					self._period_parametrisation.wrap(period),
					self._period.shape)
			)

		return new_kernel
