from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from equinox import error_if, filter_jit
from jax import Array

from .WrapperKernel import WrapperKernel


class ActiveDimsKernel(WrapperKernel):
	"""
	Wrapper kernel to select active dimensions from the inputs before passing them to the inner kernel.

	NOTE: This kernel *must* be the outer-most kernel (aka it shouldn't be wrapped inside another one)
	If you use a kernel that has HPs specific to *input dimensions* (like an ARDKernel), make sure you instantiate it
	with HPs only for the active dimensions. For example, on inputs of dimension 5 with 3 active dimensions:

	```
	# First, define ARD
	length_scales = jnp.array([1.0, 0.5, 2.0])  # Defined only on 3 dims, as we later use ARD!
	ard_kernel = ARDKernel(base_kernel, length_scales=length_scales)

	# ActiveDims must always be the outer-most kernel
	active_dims = jnp.array([0, 2, 4])
	active_kernel = ActiveDimsKernel(ard_kernel, active_dims=active_dims)
	```
	"""

	active_dims: Tuple[int, ...] = eqx.field(static=True)

	def __init__(self, inner_kernel, active_dims: Tuple[int, ...], **kwargs):
		"""
		:param inner_kernel: the kernel to wrap, must be an instance of AbstractKernel
		:param active_dims: the indices of the active dimensions to select from the inputs (1D array of integers)
		"""
		super().__init__(inner_kernel=inner_kernel, **kwargs)
		self.active_dims = tuple(int(dim) for dim in active_dims)

	@filter_jit
	def __call__(self, x1: Array, x2: None | Array = None) -> Array:
		x1 = error_if(x1, jnp.any(jnp.array(self.active_dims) >= x1.shape[-1]),
		              "active_dims contains indices out of bounds for x1.")

		if x2 is None:
			x2 = x1

		return self.inner_kernel(x1[..., self.active_dims], x2[..., self.active_dims])
