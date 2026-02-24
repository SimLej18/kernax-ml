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


class OperatorModule(AbstractModule):
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

	def replace(self, **kwargs):
		operator_kwargs = {}
		param_kwargs = {}

		for k, v in kwargs.items():
			if k in ["left", "right"]:
				operator_kwargs[k] = v
			else:
				param_kwargs[k] = v

		result = self

		if operator_kwargs:
			result = super().replace(**operator_kwargs)

		if param_kwargs:
			try:
				new_left = result.left.replace(**param_kwargs)
			except (AttributeError, TypeError):
				new_left = result.left

			try:
				new_right = result.right.replace(**param_kwargs)
			except (AttributeError, TypeError):
				new_right = result.right

			result = eqx.tree_at(lambda s: (s.left, s.right), result, (new_left, new_right))

		return result
