from __future__ import annotations
from typing import Callable
import equinox as eqx
from jax import Array
from jax import numpy as jnp
from .StationaryKernel import AbstractStationaryKernel
from ..distances import squared_euclidean_distance
from ..engines import AbstractEngine, DenseEngine
from ..parametrisations import AbstractParametrisation, LogExpParametrisation


class FeatureKernel(AbstractStationaryKernel):
	"""
	Feature Kernel with multiple positive-constrained length scales and variances.
	"""

	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	_length_scale_parametrisation: AbstractParametrisation = eqx.field()
	_length_scale_u_parametrisation: AbstractParametrisation = eqx.field()
	_variance_parametrisation: AbstractParametrisation = eqx.field()
	_length_scale: Array = eqx.field(converter=jnp.asarray)
	_length_scale_u: Array = eqx.field(converter=jnp.asarray)
	_variance: Array = eqx.field(converter=jnp.asarray)

	@property
	def length_scale(self) -> Array:
		return self._length_scale_parametrisation.unwrap(self._length_scale)

	@property
	def length_scale_u(self) -> Array:
		return self._length_scale_u_parametrisation.unwrap(self._length_scale_u)

	@property
	def variance(self) -> Array:
		return self._variance_parametrisation.unwrap(self._variance)

	def __init__(self,
	             length_scale: float | Array,
	             length_scale_u: float | Array,
	             variance: float | Array,
	             length_scale_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             length_scale_u_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             variance_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             distance_function: Callable = squared_euclidean_distance,
	             engine: AbstractEngine = DenseEngine):
		length_scale = jnp.asarray(length_scale)
		length_scale_u = jnp.asarray(length_scale_u)
		variance = jnp.asarray(variance)

		if jnp.any(length_scale <= 0):
			raise ValueError("`length_scale` must be positive.")
		if jnp.any(length_scale_u <= 0):
			raise ValueError("`length_scale_u` must be positive.")
		if jnp.any(variance <= 0):
			raise ValueError("`variance` must be positive.")

		self.distance_function = distance_function
		self._length_scale_parametrisation = length_scale_parametrisation
		self._length_scale_u_parametrisation = length_scale_u_parametrisation
		self._variance_parametrisation = variance_parametrisation
		self._length_scale = self._length_scale_parametrisation.wrap(length_scale)
		self._length_scale_u = self._length_scale_u_parametrisation.wrap(length_scale_u)
		self._variance = self._variance_parametrisation.wrap(variance)
		self.engine = engine

	def pairwise(self, x1: Array, x2: Array) -> Array:
		kern_length_scale = jnp.broadcast_to(self.length_scale, (2,)) if self.length_scale.ndim == 0 else self.length_scale
		kern_variance = jnp.broadcast_to(self.variance, (2,)) if self.variance.ndim == 0 else self.variance

		sigma_diag = kern_length_scale[0] + kern_length_scale[1] + self.length_scale_u
		sigma_det = jnp.prod(sigma_diag)

		quadratic_form = self.distance_function(x1, x2) / sigma_diag

		return (kern_variance[0] * kern_variance[1]
		        / (((2 * jnp.pi)**(len(x1) / 2)) * jnp.sqrt(sigma_det))
		        * jnp.exp(-0.5 * quadratic_form))

	def replace(self,
	            length_scale: None | float | Array = None,
	            length_scale_u: None | float | Array = None,
	            variance: None | float | Array = None,
	            **kwargs) -> FeatureKernel:
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

		if length_scale_u is not None:
			length_scale_u = jnp.asarray(length_scale_u)

			if jnp.any(length_scale_u <= 0):
				raise ValueError("`length_scale_u` must be positive.")

			new_kernel = eqx.tree_at(
				lambda k: k._length_scale_u,
				new_kernel,
				jnp.broadcast_to(
					self._length_scale_u_parametrisation.wrap(length_scale_u),
					self._length_scale_u.shape)
			)

		if variance is not None:
			variance = jnp.asarray(variance)

			if jnp.any(variance <= 0):
				raise ValueError("`variance` must be positive.")

			new_kernel = eqx.tree_at(
				lambda k: k._variance,
				new_kernel,
				jnp.broadcast_to(
					self._variance_parametrisation.wrap(variance),
					self._variance.shape)
			)

		return new_kernel
