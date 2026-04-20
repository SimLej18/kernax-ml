from __future__ import annotations
import equinox as eqx
from ..module import AbstractModule


class AbstractWrapperModule(AbstractModule):
	"""Base class for modules that wrap another module and transform its output."""
	inner: eqx.AbstractVar[AbstractModule]

	def replace(self, inner: AbstractModule | None = None, **kwargs) -> AbstractWrapperModule:
		if inner is not None:
			return eqx.tree_at(lambda m: m.inner, self, inner.replace(**kwargs))  # Still broadcast other params to new inner

		# Broadcast replace to inner module
		return eqx.tree_at(lambda m: m.inner, self, self.inner.replace(**kwargs))
