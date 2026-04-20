from __future__ import annotations
from typing import Tuple, Iterable
import equinox as eqx
from equinox import field, filter_jit
from jax import Array
from ..module import AbstractModule
from .WrapperModule import AbstractWrapperModule


class ActiveDimsModule(AbstractWrapperModule):
	"""
	Wrapper module to select active dimensions from the inputs before passing them to the inner module.

	NOTE: This module *must* be the outer-most wrapper (aka it shouldn't be wrapped inside another one).
	If you use a kernel that has HPs specific to *input dimensions* (like an ARDKernel), make sure you
	instantiate it with HPs only for the active dimensions.
	"""

	active_dims: Tuple[int, ...] = field(static=True)

	def __init__(self, inner: AbstractModule, active_dims: Iterable[int]):
		self.inner = inner
		self.active_dims = tuple(int(dim) for dim in active_dims)

	@filter_jit
	def __call__(self, x1: Array, x2: Array | None = None) -> Array:
		assert x1.shape[-1] >= max(self.active_dims), "active_dims contains indices out of bounds for x1"

		if x2 is None:
			return self.inner(x1[..., self.active_dims])
		return self.inner(x1[..., self.active_dims], x2[..., self.active_dims])

	def replace(self, active_dims: Iterable[int] | None, **kwargs):
		if active_dims is not None:
			raise ValueError(
				"`active_dims` is a static field and cannot be mutated for ActiveDimsModule. "
				"Initialise a new module instance instead.")
