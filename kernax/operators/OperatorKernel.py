import equinox as eqx

from ..AbstractKernel import AbstractKernel
from ..other.ConstantKernel import ConstantKernel


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
