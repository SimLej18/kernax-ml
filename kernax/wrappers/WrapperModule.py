from __future__ import annotations
import equinox as eqx
from ..module import AbstractModule


class AbstractWrapperModule(AbstractModule):
	"""Base class for modules that wrap another module and transform its output."""
	inner: eqx.AbstractVar[AbstractModule]

	def replace(self, **kwargs) -> AbstractWrapperModule:
		# Broadcast replace to inner module
		return eqx.tree_at(lambda k: k.inner, self, self.inner.replace(**kwargs))
