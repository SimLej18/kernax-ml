from __future__ import annotations
import equinox as eqx
from jax import Array
from jax import numpy as jnp
from ..AbstractKernel import AbstractKernel
from ..engines import AbstractEngine, DenseEngine
from ..utils import format_jax_array


class ConstantKernel(AbstractKernel):
	"""
	Constant Kernel — returns a constant value regardless of inputs.
	No positivity constraint: value can be any real number.
	"""

	engine: AbstractEngine = eqx.field(static=True)
	value: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, value: float | Array = 1.0, engine: AbstractEngine = DenseEngine):
		self.value = jnp.asarray(value)
		self.engine = engine

	def pairwise(self, x1: Array, x2: Array) -> Array:
		return self.value

	def replace(self, value: None | float | Array = None, **kwargs) -> ConstantKernel:
		if value is None:
			return self
		return eqx.tree_at(
			lambda k: k.value,
			self,
			jnp.broadcast_to(jnp.asarray(value), self.value.shape)
		)

	def __str__(self):
		return format_jax_array(self.value)
