from __future__ import annotations
from typing import Callable
import equinox as eqx
from jax import Array
from jax import numpy as jnp
from .StationaryKernel import AbstractStationaryKernel
from ..distances import euclidean_distance
from ..engines import AbstractEngine, DenseEngine
from ..parametrisations import AbstractParametrisation, LogExpParametrisation


class Matern32Kernel(AbstractStationaryKernel):
	"""Matern 3/2 Kernel"""

	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	_length_scale_parametrisation: AbstractParametrisation = eqx.field()
	_length_scale: Array = eqx.field(converter=jnp.asarray)

	@property
	def length_scale(self) -> Array:
		return self._length_scale_parametrisation.unwrap(self._length_scale)

	def __init__(self,
	             length_scale: float | Array,
	             length_scale_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             distance_function: Callable = euclidean_distance,
	             engine: AbstractEngine = DenseEngine):
		length_scale = jnp.asarray(length_scale)
		if jnp.any(length_scale <= 0):
			raise ValueError("`length_scale` must be positive.")

		self.distance_function = distance_function
		self._length_scale_parametrisation = length_scale_parametrisation
		self._length_scale = self._length_scale_parametrisation.wrap(length_scale)
		self.engine = engine

	def pairwise(self, x1: Array, x2: Array) -> Array:
		r = self.distance_function(x1, x2)
		sqrt3_r_div_l = (jnp.sqrt(3) * r) / self.length_scale
		return (1.0 + sqrt3_r_div_l) * jnp.exp(-sqrt3_r_div_l)

	def spectral_density(self, w: Array) -> Array:
		d = w.shape[-1]
		sq = jnp.sum(w ** 2, axis=-1)
		l = self.length_scale
		const = 6 * jnp.sqrt(3.0) * 2 ** d * jnp.pi ** ((d - 1) / 2) * jnp.exp(
			gammaln((d + 3) / 2)) / l ** 3
		return const * (3 / l ** 2 + sq) ** (-(d + 3) / 2)

	def replace(self, length_scale: None | float | Array = None, **kwargs) -> Matern32Kernel:
		if length_scale is None:
			return self

		length_scale = jnp.asarray(length_scale)
		if jnp.any(length_scale <= 0):
			raise ValueError("`length_scale` must be positive.")

		return eqx.tree_at(
			lambda k: k._length_scale,
			self,
			jnp.broadcast_to(
				self._length_scale_parametrisation.wrap(length_scale),
				self._length_scale.shape)
		)
