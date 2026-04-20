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


class RationalQuadraticKernel(AbstractStationaryKernel):
	"""Rational Quadratic Kernel"""

	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	_length_scale_parametrisation: AbstractParametrisation = eqx.field()
	_alpha_parametrisation: AbstractParametrisation = eqx.field()
	_length_scale: Array = eqx.field(converter=jnp.asarray)
	_alpha: Array = eqx.field(converter=jnp.asarray)

	@property
	def length_scale(self) -> Array:
		return self._length_scale_parametrisation.unwrap(self._length_scale)

	@property
	def alpha(self) -> Array:
		return self._alpha_parametrisation.unwrap(self._alpha)

	def __init__(self,
	             length_scale: float | Array,
	             alpha: float | Array,
	             length_scale_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             alpha_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             distance_function: Callable = squared_euclidean_distance,
	             engine: AbstractEngine = DenseEngine):
		length_scale = jnp.asarray(length_scale)
		alpha = jnp.asarray(alpha)

		if jnp.any(length_scale <= 0):
			raise ValueError("`length_scale` must be positive.")
		if jnp.any(alpha <= 0):
			raise ValueError("`alpha` must be positive.")

		self.distance_function = distance_function
		self._length_scale_parametrisation = length_scale_parametrisation
		self._alpha_parametrisation = alpha_parametrisation
		self._length_scale = self._length_scale_parametrisation.wrap(length_scale)
		self._alpha = self._alpha_parametrisation.wrap(alpha)
		self.engine = engine

	@filter_jit
	def pairwise(self, x1: Array, x2: Array) -> Array:
		squared_dist = self.distance_function(x1, x2)
		base = 1 + squared_dist / (2 * self.alpha * self.length_scale**2)
		return jnp.power(base, -self.alpha)

	def replace(self,
	            length_scale: None | float | Array = None,
	            alpha: None | float | Array = None,
	            **kwargs) -> RationalQuadraticKernel:
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

		if alpha is not None:
			alpha = jnp.asarray(alpha)

			if jnp.any(alpha <= 0):
				raise ValueError("`alpha` must be positive.")

			new_kernel = eqx.tree_at(
				lambda k: k._alpha,
				new_kernel,
				jnp.broadcast_to(
					self._alpha_parametrisation.wrap(alpha),
					self._alpha.shape)
			)

		return new_kernel
