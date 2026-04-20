"""
Superclass for kernels and mean functions, extending Equinox's Module with operators and formatting.
"""
from __future__ import annotations
from abc import abstractmethod
import equinox as eqx
from jax import Array


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

		# 2. Public Array fields not covered by properties
		for key, value in self.__dict__.items():
			if not key.startswith('_') and key not in seen and isinstance(value, Array):
				parts.append(f'{key}={format_jax_array(value)}')
				seen.add(key)

		# 3. Public int/float/str fields (e.g. degree in PolynomialKernel)
		for key, value in self.__dict__.items():
			if not key.startswith('_') and key not in seen and isinstance(value, (int, float, str)):
				parts.append(f'{key}={value}')

		return f"{self.__class__.__name__}({', '.join(parts)})"