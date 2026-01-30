from jax import Array, jit

from .WrapperKernel import WrapperKernel


class NegKernel(WrapperKernel):
	@jit
	def __call__(self, x1: Array, x2: None | Array = None) -> Array:
		if x2 is None:
			x2 = x1

		return -self.inner_kernel(x1, x2)

	def __str__(self):
		return f"- {self.inner_kernel}"
