import equinox as eqx
from jax import jit
from jax import numpy as jnp

from kernax import AbstractKernel, ConstantKernel


class OperatorKernel(AbstractKernel):
	"""Class for kernels that apply some operation on the output of two kernels."""

	left_kernel: AbstractKernel = eqx.field()
	right_kernel: AbstractKernel = eqx.field()

	def __init__(self, left_kernel, right_kernel):
		"""
		Instantiates a sum kernel with the given kernels.

		:param right_kernel: the right kernel to sum
		:param left_kernel: the left kernel to sum
		"""
		# If any of the provided arguments are not kernels, we try to convert them to ConstantKernels
		if not isinstance(left_kernel, AbstractKernel):
			left_kernel = ConstantKernel(value=left_kernel)
		if not isinstance(right_kernel, AbstractKernel):
			right_kernel = ConstantKernel(value=right_kernel)

		super().__init__()
		self.left_kernel = left_kernel
		self.right_kernel = right_kernel


class SumKernel(OperatorKernel):
	"""Sum kernel that sums the outputs of two kernels."""

	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return self.left_kernel(x1, x2) + self.right_kernel(x1, x2)

	def __str__(self):
		# If the right kernel is a NegKernel, we format it as a subtraction
		if self.right_kernel.__class__.__name__ == "NegKernel":
			return f"{self.left_kernel} - {self.right_kernel.inner_kernel}"
		return f"{self.left_kernel} + {self.right_kernel}"


class ProductKernel(OperatorKernel):
	"""Product kernel that multiplies the outputs of two kernels."""

	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return self.left_kernel(x1, x2) * self.right_kernel(x1, x2)

	def __str__(self):
		# If the right kernel is a NegKernel, we add parentheses around it
		if self.right_kernel.__class__.__name__ == "NegKernel":
			return f"{self.left_kernel} * ({self.right_kernel})"
		return f"{self.left_kernel} * {self.right_kernel}"
