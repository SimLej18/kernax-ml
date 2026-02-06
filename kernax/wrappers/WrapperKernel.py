import equinox as eqx

from ..AbstractKernel import AbstractKernel
from ..other.ConstantKernel import ConstantKernel


class WrapperKernel(AbstractKernel):
	"""Class for kernels that perform some operation on the output of another "inner" kernel."""

	inner_kernel: AbstractKernel = eqx.field()

	def __init__(self, inner_kernel=None, **kwargs):
		"""
		Instantiates a wrapper kernel with the given inner kernel.

		:param inner_kernel: the inner kernel to wrap
		"""
		super().__init__(**kwargs)

		# If the inner kernel is not a kernel, we try to convert it to a ConstantKernel
		if not isinstance(inner_kernel, AbstractKernel):
			inner_kernel = ConstantKernel(value=inner_kernel)

		self.inner_kernel = inner_kernel

	def replace(self, **kwargs):
		"""
		Replace parameters, forwarding to inner_kernel when appropriate.

		If a parameter exists on the wrapper itself, it's replaced directly.
		Otherwise, the replacement is forwarded to inner_kernel.
		"""
		# Separate wrapper's own parameters from inner_kernel parameters
		wrapper_kwargs = {}
		inner_kwargs = {}

		for k, v in kwargs.items():
			# Check if it's a direct attribute of this wrapper (excluding inner_kernel)
			if k == "inner_kernel" or (
				hasattr(self, k) and k not in ["inner_kernel"] and hasattr(type(self), k)
			):
				wrapper_kwargs[k] = v
			else:
				inner_kwargs[k] = v

		# Apply changes to wrapper's own parameters
		result = self
		if wrapper_kwargs:
			result = super().replace(**wrapper_kwargs)

		# Forward inner_kernel parameter changes
		if inner_kwargs:
			new_inner = result.inner_kernel.replace(**inner_kwargs)
			result = eqx.tree_at(lambda s: s.inner_kernel, result, new_inner)

		return result
