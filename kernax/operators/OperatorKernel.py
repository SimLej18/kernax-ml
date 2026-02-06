import equinox as eqx

from ..AbstractKernel import AbstractKernel
from ..other.ConstantKernel import ConstantKernel


class OperatorKernel(AbstractKernel):
	"""Class for kernels that apply some operation on the output of two kernels."""

	left_kernel: AbstractKernel = eqx.field()
	right_kernel: AbstractKernel = eqx.field()

	def __init__(self, left_kernel, right_kernel, **kwargs):
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

		super().__init__(**kwargs)
		self.left_kernel = left_kernel
		self.right_kernel = right_kernel

	def replace(self, **kwargs):
		"""
		Replace parameters in both left and right kernels.

		If replacing left_kernel or right_kernel directly, those are updated.
		Otherwise, parameters are forwarded to both sub-kernels (ignoring if not applicable).
		"""
		# Separate direct kernel replacements from parameter modifications
		operator_kwargs = {}
		param_kwargs = {}

		for k, v in kwargs.items():
			if k in ["left_kernel", "right_kernel"]:
				operator_kwargs[k] = v
			else:
				param_kwargs[k] = v

		# Start with current kernel
		result = self

		# Apply direct kernel replacements
		if operator_kwargs:
			result = super().replace(**operator_kwargs)

		# Apply parameter changes to both left and right kernels
		if param_kwargs:
			# Try to apply to left kernel (may silently fail if parameter doesn't exist)
			try:
				new_left = result.left_kernel.replace(**param_kwargs)
			except (AttributeError, TypeError):
				new_left = result.left_kernel

			# Try to apply to right kernel (may silently fail if parameter doesn't exist)
			try:
				new_right = result.right_kernel.replace(**param_kwargs)
			except (AttributeError, TypeError):
				new_right = result.right_kernel

			# Update both kernels
			result = eqx.tree_at(
				lambda s: (s.left_kernel, s.right_kernel), result, (new_left, new_right)
			)

		return result
