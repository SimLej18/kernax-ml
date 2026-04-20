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


class AffineKernel(AbstractDotProductKernel):
	"""
	Affine Kernel, corresponding to the formula:
	k(x, x') = slope_var * (x - offset).T @ (x' - offset)

	Note: In GPs, samples/posteriors from this kernel will always cross the points (offset, 0).
	If you want to add uncertainty as to where the crossing point is, you should add a
	ConstantKernel to the AffineKernel (i.e. use a SumKernel) where the constant value
	represents the variance at the crossing point.
	"""
	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	_slope_var_parametrisation: AbstractParametrisation = eqx.field()
	_slope_var: Array = eqx.field(converter=jnp.asarray)
	offset: Array = eqx.field(converter=jnp.asarray)

	@property
	def slope_var(self) -> Array:
		return self._slope_var_parametrisation.unwrap(self._slope_var)

	def __init__(self,
	             slope_var: float | Array,
	             offset: float | Array,
	             slope_var_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             distance_function: Callable = dot_product,
	             engine: AbstractEngine = DenseEngine):
		"""
		Initialize the Affine kernel.

		Args:
			slope_var: Slope variance. Controls the slope. Must be positive.
			offset: Input offset. Determines the crossing point.
		"""
		slope_var = jnp.asarray(slope_var)
		if jnp.any(slope_var <= 0):
			raise ValueError("`slope_var` must be positive.")

		self.distance_function = distance_function
		self._slope_var_parametrisation = slope_var_parametrisation
		self._slope_var = self._slope_var_parametrisation.wrap(slope_var)
		self.offset = jnp.asarray(offset)
		self.engine = engine

	@filter_jit
	def pairwise(self, x1: Array, x2: Array) -> Array:
		return self.slope_var * self.distance_function(x1 - self.offset, x2 - self.offset)

	def replace(self,
	            slope_var: None | float | Array = None,
	            offset: None | float | Array = None,
	            **kwargs) -> AffineKernel:
		new_kernel = self

		if slope_var is not None:
			slope_var = jnp.asarray(slope_var)

			if jnp.any(slope_var <= 0):
				raise ValueError("`slope_var` must be positive.")

			new_kernel = eqx.tree_at(
				lambda k: k._slope_var,
				new_kernel,
				jnp.broadcast_to(
					self._slope_var_parametrisation.wrap(slope_var),
					self._slope_var.shape)
			)

		if offset is not None:
			new_kernel = eqx.tree_at(
				lambda k: k.offset,
				new_kernel,
				jnp.broadcast_to(jnp.asarray(offset), self.offset.shape)
			)

		return new_kernel
