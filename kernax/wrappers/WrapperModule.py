from __future__ import annotations

from typing import Generic

import equinox as eqx

from ..module import AbstractModule, ModuleT


class AbstractWrapperModule(AbstractModule, Generic[ModuleT]):
	"""Base class for modules that wrap another module and transform its output."""
	inner: eqx.AbstractVar[ModuleT]

	def replace(self, inner: ModuleT | None = None, **kwargs) -> AbstractWrapperModule[ModuleT]:
		if inner is not None:
			return eqx.tree_at(lambda m: m.inner, self, inner.replace(**kwargs))  # Still broadcast other params to new inner

		# Broadcast replace to inner module
		return eqx.tree_at(lambda m: m.inner, self, self.inner.replace(**kwargs))
