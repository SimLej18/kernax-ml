from __future__ import annotations
import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp
from .DotProductKernel import AbstractDotProductKernel
from ...engines import AbstractEngine, DenseEngine
from ...parametrisations import AbstractParametrisation, LogExpParametrisation
from ...distances import dot_product


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
	_distance_function: Callable = eqx.field(static=True)
	_slope_var_parametrisation: AbstractParametrisation = eqx.field(static=True)
	_slope_var: Array = eqx.field(converter=jnp.asarray)
	offset: Array = eqx.field(converter=jnp.asarray)

	@property
	def slope_var(self):
		return self._slope_var_parametrisation.unwrap(self._slope_var)

	def __init__(self,
	             slope_var: float | Array,
	             offset: float | Array,
	             slope_var_parametrisation: AbstractParametrisation = LogExpParametrisation,
	             distance_function: Callable = dot_product,
	             engine: AbstractEngine = DenseEngine
	             ):
		"""
		Initialize the Linear kernel.

		Args:
			slope_var: Slope variance. Controls the slope. Must be non-negative.
			offset: Input offset. Determines the crossing point.
		"""
		# Assert slope_var is positive
		slope_var = jnp.asarray(slope_var)
		slope_var = eqx.error_if(slope_var, jnp.any(self.slope_var < 0),  # FIXME: we don't have to use error_if here, we are not inside jit
		                         "`slope_var` must be non-negative.")

		self._distance_function = distance_function
		self._slope_var_parametrisation = slope_var_parametrisation
		self._slope_var = self._slope_var_parametrisation.wrap(slope_var)
		self.offset = offset
		self.engine = engine

	@filter_jit
	def pairwise_(self, x1: Array, x2: Array) -> Array:
		"""
		Compute the affine kernel covariance value between two vectors.

		:param x1: scalar array.
		:param x2: scalar array.
		:return: scalar array (covariance value).
		"""
		# Compute the dot product of the shifted vectors
		return self.slope_var * self._distance_function(x1 - self.offset, x2 - self.offset)

	def replace(self,
	            slope_var: None | float | Array = None,
	            offset: None | float | Array = None,
	            **kwargs) -> AffineKernel:
		new_kernel = self

		if slope_var is not None:
			new_kernel =  eqx.tree_at(
				lambda k: k._slope_var,
				new_kernel,
				jnp.broadcast_to(
					self._slope_var_parametrisation.wrap(jnp.asarray(slope_var)),
					self._slope_var.shape)
			)

		if offset is not None:
			new_kernel = eqx.tree_at(
				lambda k: k.offset,
				new_kernel,
				jnp.broadcast_to(
					jnp.asarray(offset),
					self._slope_var.shape),
			)

		return new_kernel
