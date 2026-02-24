from __future__ import annotations

import equinox as eqx

from ..module import AbstractModule
from ..other.ConstantKernel import ConstantKernel


class WrapperModule(AbstractModule):
	"""Base class for modules that wrap another module and transform its output."""

	inner: AbstractModule = eqx.field()

	def __init__(self, inner=None, **kwargs):
		if not isinstance(inner, AbstractModule):
			inner = ConstantKernel(value=inner)

		super().__init__(**kwargs)
		self.inner = inner

	def replace(self, **kwargs):
		wrapper_kwargs = {}
		inner_kwargs = {}

		for k, v in kwargs.items():
			if k == "inner" or (hasattr(self, k) and k != "inner" and hasattr(type(self), k)):
				wrapper_kwargs[k] = v
			else:
				inner_kwargs[k] = v

		result = self
		if wrapper_kwargs:
			result = super().replace(**wrapper_kwargs)

		if inner_kwargs:
			new_inner = result.inner.replace(**inner_kwargs)
			result = eqx.tree_at(lambda s: s.inner, result, new_inner)

		return result