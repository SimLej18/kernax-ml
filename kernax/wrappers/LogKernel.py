import jax.numpy as jnp
from jax import Array, jit

from .WrapperKernel import WrapperKernel


class LogKernel(WrapperKernel):
	"""
	Kernel that applies the logarithm operator to the output of another kernel.
	"""

	@jit
	def __call__(self, x1: Array, x2: None | Array = None) -> Array:
		if x2 is None:
			x2 = x1

		return jnp.log(self.inner_kernel(x1, x2))

	def __str__(self):
		return f"Log({self.inner_kernel})"
