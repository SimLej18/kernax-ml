from __future__ import annotations

from typing import Generic

import jax.numpy as jnp
from jax import Array

from ..module import AbstractModule, ModuleT
from ..other.ConstantKernel import ConstantKernel
from .WrapperModule import AbstractWrapperModule


class ExpModule(AbstractWrapperModule[ModuleT], Generic[ModuleT]):
	"""Module that applies the exponential function to the output of another module."""
	inner: ModuleT

	def __init__(self, inner=None):
		if not isinstance(inner, AbstractModule):
			inner = ConstantKernel(value=inner)

		self.inner = inner

	def __call__(self, x1: Array, x2: Array | None = None) -> Array:
		if x2 is None:
			return jnp.exp(self.inner(x1))
		return jnp.exp(self.inner(x1, x2))

	def spectral_density(self, w: Array) -> Array:
		raise NotImplementedError(
			"`ExpModule` has no spectral density in closed form: the Fourier transform of "
			"`exp(k)` is not a function of the transform of `k`. Reach into `inner` "
			"explicitly for the wrapped kernel's own spectral density."
		)
