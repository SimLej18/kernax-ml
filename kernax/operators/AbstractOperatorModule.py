from __future__ import annotations

import equinox as eqx

from ..module import AbstractModule


def _to_constant(value, reference_module):
	"""Convert a non-Module value to the appropriate Constant type based on the other operand."""
	from ..AbstractKernel import AbstractKernel
	from ..AbstractMean import AbstractMean

	if isinstance(reference_module, AbstractMean):
		from ..means.ConstantMean import ConstantMean
		return ConstantMean(constant=value)
	else:
		from ..other.ConstantKernel import ConstantKernel
		return ConstantKernel(value=value)


class AbstractOperatorModule(AbstractModule):
	"""Base class for modules that apply an operation on the outputs of two sub-modules."""

	left: AbstractModule = eqx.field()
	right: AbstractModule = eqx.field()

	def __init__(self, left, right, **kwargs):
		if not isinstance(left, AbstractModule):
			left = _to_constant(left, right)
		if not isinstance(right, AbstractModule):
			right = _to_constant(right, left)

		super().__init__(**kwargs)
		self.left = left
		self.right = right

	def replace(self,
	            left: AbstractModule | None = None,
	            right: AbstractModule | None = None,
	            **kwargs):
		new_module = self

		if left is not None:
			# Still broadcast other params to new left
			new_module = eqx.tree_at(lambda m: m.left, new_module, left.replace(**kwargs))
		else:
			# Only broadcast to current left module
			new_module = eqx.tree_at(lambda m: m.left, new_module, new_module.left.replace(**kwargs))

		if right is not None:
			# Still broadcast other params to new right
			new_module = eqx.tree_at(lambda m: m.right, new_module, right.replace(**kwargs))
		else:
			new_module = eqx.tree_at(lambda m: m.right, new_module, new_module.right.replace(**kwargs))

		return new_module
