from __future__ import annotations

from typing import Any, Generic

import equinox as eqx

from ..module import AbstractModule, ModuleT, has_stored_hp


class AbstractWrapperModule(AbstractModule, Generic[ModuleT]):
	"""Base class for modules that wrap another module and transform its output.

	Hyperparameter access is transparent through the wrapping: any hyperparameter stored in
	the wrapped tree can be read from the outermost module, at any nesting depth::

	    kernel = ICMKernel(ActiveDimsModule(SEKernel(1.0), [0]), n_outputs=3, n_latent=3)
	    kernel.length_scale        # 1.0 -- the SEKernel's, reached through two wrappers

	Only *stored* hyperparameters are forwarded (see
	:func:`~kernax.module.has_stored_hp`); computed quantities are not. A wrapper exists
	precisely to change what its inner module computes, so relaying the inner module's
	answer would be wrong: ``ICMKernel`` has no scalar spectral density, and an
	``ARDKernel``'s is not its inner kernel's. A wrapper that can transform such a quantity
	defines it explicitly (:meth:`~kernax.wrappers.ARDKernel.ARDKernel.spectral_density`);
	one that cannot raises ``NotImplementedError``.

	Operators (:class:`~kernax.operators.OperatorModule.AbstractOperatorModule`) do not
	forward: with two operands, a name carried by both has no unambiguous owner. Reach into
	``left``/``right`` explicitly there.
	"""
	inner: eqx.AbstractVar[ModuleT]

	def __getattr__(self, name: str) -> Any:
		if name.startswith("_"):
			raise AttributeError(name)

		try:
			inner = object.__getattribute__(self, "inner")
		except AttributeError:  # accessed before `__init__` assigned `inner`
			raise AttributeError(name) from None

		if not has_stored_hp(inner, name):
			raise AttributeError(
				f"'{type(self).__name__}' object has no attribute '{name}', and "
				f"'{type(inner).__name__}' stores no hyperparameter of that name. Only "
				f"stored hyperparameters are forwarded through a wrapper; computed "
				f"quantities must be defined by the wrapper itself."
			)

		return getattr(inner, name)

	def replace(self, inner: ModuleT | None = None, **kwargs) -> AbstractWrapperModule[ModuleT]:
		if inner is not None:
			return eqx.tree_at(lambda m: m.inner, self, inner.replace(**kwargs))  # Still broadcast other params to new inner

		# Broadcast replace to inner module
		return eqx.tree_at(lambda m: m.inner, self, self.inner.replace(**kwargs))
