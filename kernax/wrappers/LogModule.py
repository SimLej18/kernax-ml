from __future__ import annotations

from typing import Generic

import jax.numpy as jnp
from jax import Array

from ..module import AbstractModule, ModuleT
from ..other.ConstantKernel import ConstantKernel
from .WrapperModule import AbstractWrapperModule


class LogModule(AbstractWrapperModule[ModuleT], Generic[ModuleT]):
	"""Module that applies the logarithm function to the output of another module."""
	inner: ModuleT

	def __init__(self, inner=None):
		if not isinstance(inner, AbstractModule):
			inner = ConstantKernel(value=inner)

		self.inner = inner

	def __call__(self, x1: Array, x2: Array | None = None) -> Array:
		if x2 is None:
			return jnp.log(self.inner(x1))
		return jnp.log(self.inner(x1, x2))
