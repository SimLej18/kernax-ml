from __future__ import annotations
import jax.numpy as jnp
from jax import Array
from .WrapperModule import AbstractWrapperModule
from ..module import AbstractModule
from ..other.ConstantKernel import ConstantKernel


class ExpModule(AbstractWrapperModule):
	"""Module that applies the exponential function to the output of another module."""
	inner: AbstractModule

	def __init__(self, inner=None):
		if not isinstance(inner, AbstractModule):
			inner = ConstantKernel(value=inner)

		self.inner = inner

	def __call__(self, x1: Array, x2: Array | None = None) -> Array:
		if x2 is None:
			return jnp.exp(self.inner(x1))
		return jnp.exp(self.inner(x1, x2))

	def __str__(self):
		return f"Exp({self.inner})"
