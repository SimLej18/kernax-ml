import equinox as eqx
import jax.numpy as jnp
from jax import Array, jit

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

	active_dims: Array = eqx.field(static=True, converter=jnp.asarray)

	def __init__(self, inner_kernel, active_dims):
		"""
		:param inner_kernel: the kernel to wrap, must be an instance of AbstractKernel
		:param active_dims: the indices of the active dimensions to select from the inputs (1D array of integers)
		"""
		super().__init__(inner_kernel=inner_kernel)
		self.active_dims = active_dims

	@jit
	def __call__(self, x1: Array, x2: None | Array = None) -> Array:
		# TODO: add runtime error if active_dims doesn't match input dimensions
		if x2 is None:
			x2 = x1

		return self.inner_kernel(x1[..., self.active_dims], x2[..., self.active_dims])
