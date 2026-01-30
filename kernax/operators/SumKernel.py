from typing import Optional

from jax import Array, jit

from .OperatorKernel import OperatorKernel


class SumKernel(OperatorKernel):
	"""Sum kernel that sums the outputs of two kernels."""

	@jit
	def __call__(self, x1: Array, x2: Optional[Array] = None) -> Array:
		if x2 is None:
			x2 = x1

		return self.left_kernel(x1, x2) + self.right_kernel(x1, x2)

	def __str__(self):
		# If the right kernel is a NegKernel, we format it as a subtraction
		if self.right_kernel.__class__.__name__ == "NegKernel":
			return f"{self.left_kernel} - {self.right_kernel.inner_kernel}"  # type: ignore[attr-defined]
		return f"{self.left_kernel} + {self.right_kernel}"
