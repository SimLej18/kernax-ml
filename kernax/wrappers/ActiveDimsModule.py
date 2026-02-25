from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
from equinox import error_if, field, filter_jit
from jax import Array

from .WrapperModule import WrapperModule


class ActiveDimsModule(WrapperModule):
	"""
	Wrapper module to select active dimensions from the inputs before passing them to the inner module.

	NOTE: This module *must* be the outer-most wrapper (aka it shouldn't be wrapped inside another one).
	If you use a kernel that has HPs specific to *input dimensions* (like an ARDKernel), make sure you
	instantiate it with HPs only for the active dimensions.
	"""

	active_dims: Tuple[int, ...] = field(static=True)

	def __init__(self, inner, active_dims: Tuple[int, ...], **kwargs):
		super().__init__(inner=inner, **kwargs)
		self.active_dims = tuple(int(dim) for dim in active_dims)

	def replace(self, **kwargs):
		if "active_dims" in kwargs:
			raise ValueError(
				"'active_dims' is a structural parameter of ActiveDimsModule and cannot be "
				"modified via replace(). Create a new ActiveDimsModule with the desired active_dims."
			)
		return super().replace(**kwargs)

	@filter_jit
	def __call__(self, x1: Array, x2: Optional[Array] = None) -> Array:
		x1 = error_if(x1, jnp.any(jnp.array(self.active_dims) >= x1.shape[-1]),
		              "active_dims contains indices out of bounds for x1.")

		if x2 is None:
			return self.inner(x1[..., self.active_dims])
		return self.inner(x1[..., self.active_dims], x2[..., self.active_dims])