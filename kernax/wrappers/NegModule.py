from __future__ import annotations
from equinox import filter_jit
from jax import Array
from .WrapperModule import AbstractWrapperModule
from ..module import AbstractModule
from ..other.ConstantKernel import ConstantKernel


class NegModule(AbstractWrapperModule):
	inner: AbstractModule

	def __init__(self, inner=None):
		if not isinstance(inner, AbstractModule):
			inner = ConstantKernel(value=inner)

		self.inner = inner

	@filter_jit
	def __call__(self, x1: Array, x2: Array | None = None) -> Array:
		if x2 is None:
			return -self.inner(x1)
		return -self.inner(x1, x2)

	def __str__(self):
		return f"- {self.inner}"
