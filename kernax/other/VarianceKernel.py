from __future__ import annotations
import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp
from ..AbstractKernel import AbstractKernel
from ..engines import AbstractEngine, DenseEngine
from ..parametrisations import AbstractParametrisation, LogExpParametrisation


class VarianceKernel(AbstractKernel):
	"""
	Variance kernel that returns a positive constant value everywhere.
	Used to multiply with other kernels to scale their output.
	"""

	engine: AbstractEngine = eqx.field(static=True)
	_variance_parametrisation: AbstractParametrisation = eqx.field()
	_variance: Array = eqx.field(converter=jnp.asarray)

	@property
	def variance(self) -> Array:
		return self._variance_parametrisation.unwrap(self._variance)

	def __init__(self,
	             variance: float | Array = 1.0,
	             variance_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             engine: AbstractEngine = DenseEngine):
		variance = jnp.asarray(variance)
		if jnp.any(variance <= 0):
			raise ValueError("`variance` must be positive.")

		self._variance_parametrisation = variance_parametrisation
		self._variance = self._variance_parametrisation.wrap(variance)
		self.engine = engine

	@filter_jit
	def pairwise(self, x1: Array, x2: Array) -> Array:
		return self.variance

	def replace(self, variance: None | float | Array = None, **kwargs) -> VarianceKernel:
		if variance is None:
			return self

		variance = jnp.asarray(variance)

		if jnp.any(variance <= 0):
			raise ValueError("`variance` must be positive.")

		return eqx.tree_at(
			lambda k: k._variance,
			self,
			jnp.broadcast_to(
				self._variance_parametrisation.wrap(variance),
				self._variance.shape)
		)
