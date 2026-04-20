from __future__ import annotations
import jax.numpy as jnp
from equinox import filter_jit
from jax import Array
from .WrapperModule import AbstractWrapperModule
from ..module import AbstractModule
from ..other.ConstantKernel import ConstantKernel


class LogModule(AbstractWrapperModule):
	"""Module that applies the logarithm function to the output of another module."""
	inner: AbstractModule

	def __init__(self, inner=None):
		if not isinstance(inner, AbstractModule):
			inner = ConstantKernel(value=inner)

		self.inner = inner

	@filter_jit
	def __call__(self, x1: Array, x2: Array | None = None) -> Array:
		if x2 is None:
			return jnp.log(self.inner(x1))
		return jnp.log(self.inner(x1, x2))

	def __str__(self):
		return f"Log({self.inner})"
