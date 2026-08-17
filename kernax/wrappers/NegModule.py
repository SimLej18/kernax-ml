from __future__ import annotations

from typing import Generic

from jax import Array

from ..module import AbstractModule, ModuleT
from ..other.ConstantKernel import ConstantKernel
from .WrapperModule import AbstractWrapperModule


class NegModule(AbstractWrapperModule[ModuleT], Generic[ModuleT]):
	inner: ModuleT

	def __init__(self, inner=None):
		if not isinstance(inner, AbstractModule):
			inner = ConstantKernel(value=inner)

		self.inner = inner

	def __call__(self, x1: Array, x2: Array | None = None, **kwargs) -> Array:
		if x2 is None:
			return -self.inner(x1, **kwargs)
		return -self.inner(x1, x2, **kwargs)

	def __str__(self):
		return f"- {self.inner}"
