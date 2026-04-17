import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp
from .DotProductKernel import AbstractDotProductKernel
from ...engines import AbstractEngine, DenseEngine
from ...parametrisations import AbstractParametrisation, LogExpParametrisation
from ...distances import dot_product


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
	_distance_function: Callable = eqx.field(static=True)
	_slope_var_parametrisation: AbstractParametrisation = eqx.field(static=True)
	_slope_var: Array = eqx.field(converter=jnp.asarray)

	@property
	def slope_var(self):
		return self._slope_var_parametrisation.unwrap(self._slope_var)

	def __init__(self,
	             slope_var: float | Array,
	             slope_var_parametrisation: AbstractParametrisation = LogExpParametrisation,
	             distance_function: Callable = dot_product,
	             engine: AbstractEngine = DenseEngine
	             ):
		"""
		Initialize the Linear kernel.

		Args:
			slope_var: Slope variance. Controls the slope. Must be non-negative.
		"""
		# Assert slope_var is positive
		slope_var = jnp.asarray(slope_var)
		slope_var = eqx.error_if(slope_var, jnp.any(self.slope_var < 0),
		                         "`slope_var` must be non-negative.")

		self._distance_function = distance_function
		self._slope_var_parametrisation = slope_var_parametrisation
		self._slope_var = self._slope_var_parametrisation.wrap(slope_var)
		self.engine = engine

	@filter_jit
	def pairwise(self, x1: Array, x2: Array) -> Array:
		"""
		Compute the linear kernel covariance value between two vectors.

		:param x1: scalar array.
		:param x2: scalar array.
		:return: scalar array (covariance value).
		"""
		# Compute the dot product of the shifted vectors
		return self.slope_var * self._distance_function(x1, x2)

	def replace(self, slope_var: None|float|Array = None, **kwargs) -> LinearKernel:
		if slope_var is None:
			return self  # No change to make

		return eqx.tree_at(
			lambda k: k._slope_var,
			self,
			jnp.broadcast_to(
				self._slope_var_parametrisation.wrap(jnp.asarray(slope_var)),
				self._slope_var.shape)
		)
