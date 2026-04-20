from __future__ import annotations
from typing import Callable
import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp
from .DotProductKernel import AbstractDotProductKernel
from ..engines import AbstractEngine, DenseEngine
from ..parametrisations import AbstractParametrisation, LogExpParametrisation
from ..distances import dot_product


class LinearKernel(AbstractDotProductKernel):
	"""
	Linear Kernel, corresponding to the formula:
	k(x, x') = slope_var * x.T @ x'

	Note: In GPs, samples/posteriors from this kernel will always cross the points (0, 0).
	If you want to enable other behaviors, you can either:
	* use an AffineKernel with fixed offset
	* add a ConstantKernel to the LinearKernel (i.e. use a SumKernel) where the constant value
	represents the variance at the crossing point.
	"""
	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	_slope_var_parametrisation: AbstractParametrisation = eqx.field()
	_slope_var: Array = eqx.field(converter=jnp.asarray)

	@property
	def slope_var(self) -> Array:
		return self._slope_var_parametrisation.unwrap(self._slope_var)

	def __init__(self,
	             slope_var: float | Array,
	             slope_var_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             distance_function: Callable = dot_product,
	             engine: AbstractEngine = DenseEngine):
		"""
		Initialize the Linear kernel.

		Args:
			slope_var: Slope variance. Controls the slope. Must be positive.
		"""
		slope_var = jnp.asarray(slope_var)
		if jnp.any(slope_var <= 0):
			raise ValueError("`slope_var` must be positive.")

		self.distance_function = distance_function
		self._slope_var_parametrisation = slope_var_parametrisation
		self._slope_var = self._slope_var_parametrisation.wrap(slope_var)
		self.engine = engine

	@filter_jit
	def pairwise(self, x1: Array, x2: Array) -> Array:
		return self.slope_var * self.distance_function(x1, x2)

	def replace(self, slope_var: None | float | Array = None, **kwargs) -> LinearKernel:
		if slope_var is None:
			return self

		slope_var = jnp.asarray(slope_var)

		if jnp.any(slope_var <= 0):
			raise ValueError("`slope_var` must be positive.")

		return eqx.tree_at(
			lambda k: k._slope_var,
			self,
			jnp.broadcast_to(
				self._slope_var_parametrisation.wrap(slope_var),
				self._slope_var.shape)
		)
