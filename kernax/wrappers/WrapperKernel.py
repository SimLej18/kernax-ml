import equinox as eqx

from ..AbstractKernel import AbstractKernel
from ..other.ConstantKernel import ConstantKernel


class WrapperKernel(AbstractKernel):
	"""Class for kernels that perform some operation on the output of another "inner" kernel."""

	inner_kernel: AbstractKernel = eqx.field()

	def __init__(self, inner_kernel=None):
		"""
		Instantiates a wrapper kernel with the given inner kernel.

		:param inner_kernel: the inner kernel to wrap
		"""
		# If the inner kernel is not a kernel, we try to convert it to a ConstantKernel
		if not isinstance(inner_kernel, AbstractKernel):
			inner_kernel = ConstantKernel(value=inner_kernel)

		self.inner_kernel = inner_kernel
