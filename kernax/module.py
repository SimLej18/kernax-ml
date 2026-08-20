"""
Superclass for kernels and mean functions, extending Equinox's Module with operators and formatting.
"""
from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import TypeVar

import equinox as eqx
from jax import Array

ModuleT = TypeVar("ModuleT", bound="AbstractModule", covariant=True)
"""Type of the module(s) held by a wrapper/operator module (``inner``, ``left``/``right``).
Covariant so that, e.g., ``AbstractWrapperModule[SEKernel]`` counts as an
``AbstractWrapperModule[KernelLike]`` -- see :mod:`kernax.types`."""


class AbstractModule(eqx.Module):
	@abstractmethod
	def __call__(self: AbstractModule, *args, **kwargs) -> Array:
		raise NotImplementedError

	def __add__(self, other):
		from kernax.operators import SumModule
		return SumModule(self, other)

	def __radd__(self, other):
		from kernax.operators import SumModule
		return SumModule(other, self)

	def __sub__(self, other):
		from kernax.operators import SumModule
		from kernax.wrappers import NegModule
		return SumModule(self, NegModule(other))

	def __rsub__(self, other):
		from kernax.operators import SumModule
		from kernax.wrappers import NegModule
		return SumModule(other, NegModule(self))

	def __neg__(self):
		from kernax.wrappers import NegModule
		return NegModule(self)

	def __mul__(self, other):
		from kernax.operators import ProductModule
		return ProductModule(self, other)

	def __rmul__(self, other):
		from kernax.operators import ProductModule
		return ProductModule(other, self)

	def __str__(self):
		from kernax.utils import format_jax_array

		parts = []
		cls = type(self)
		seen: set[str] = set()

		# 1. Properties returning Arrays (expose constrained param values)
		for klass in cls.__mro__:
			for name, obj in vars(klass).items():
				if name in seen or name.startswith('_'):
					continue
				if isinstance(obj, property):
					try:
						val = getattr(self, name)
						if isinstance(val, Array):
							parts.append(f'{name}={format_jax_array(val)}')
							seen.add(name)
					except Exception:
						pass

		# 2. Public Array and Module fields not covered by properties
		for key, value in self.__dict__.items():
			if not key.startswith('_') and key not in seen:
				if isinstance(value, Array):
					parts.append(f'{key}={format_jax_array(value)}')
					seen.add(key)
				elif isinstance(value, eqx.Module):
					parts.append(f'{key}={value}')

		# 3. Public int/float/str fields (e.g. degree in PolynomialKernel)
		for key, value in self.__dict__.items():
			if not key.startswith('_') and key not in seen and isinstance(value, (int, float, str)):
				parts.append(f'{key}={value}')

		return f"{self.__class__.__name__}({', '.join(parts)})"


def has_own_attr(module: eqx.Module, name: str) -> bool:
	"""True if `name` is a field or property that `type(module)` itself declares.

	Plain `hasattr` is too permissive for wrappers, which forward any hyperparameter of
	their (possibly nested) `inner` -- `hasattr` would then report every one of `inner`'s
	hyperparameters as the wrapper's own.
	"""
	if name in {f.name for f in dataclasses.fields(type(module))}:
		return True
	return any(isinstance(vars(klass).get(name), property) for klass in type(module).__mro__)


def has_stored_hp(module: eqx.Module, name: str) -> bool:
	"""True if `name` designates a hyperparameter *stored* somewhere in `module`'s tree.

	A name is stored when a module along the `inner` chain declares it as a dataclass field,
	either directly (`ICMKernel.W`) or as the private field behind a parametrised property
	(`_length_scale`, exposed as `length_scale`).

	Quantities a module *computes* from its fields are not stored, whether they are written
	as methods (`spectral_density`) or as properties (`ICMKernel.coregionalisation`). The
	distinction is what wrapper forwarding hinges on: a wrapper does not change what a
	stored hyperparameter *is*, but it does change what its inner module *computes*, so it
	must never answer for the inner module's version of a computed quantity.
	"""
	while isinstance(module, eqx.Module):
		fields = {f.name for f in dataclasses.fields(type(module))}
		if name in fields or f"_{name}" in fields:
			return True
		if "inner" not in fields:
			return False
		module = module.inner  # type: ignore[attr-defined]
	return False
