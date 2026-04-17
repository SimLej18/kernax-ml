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
		# FIXME
		from kernax.transforms import to_constrained
		from kernax.utils import format_jax_array

		params = [f'{key}={format_jax_array(self.__getattribute__(key))}' if '_raw_' not in key
		          else (f'{key[5:]}={format_jax_array(self.__getattribute__(key[5:]))}'
		                if isinstance(value, Array)
						else f'{key[5:]}=None'
		                )
		          for key, value in self.__dict__.items()
		          if isinstance(value, Array) or (value is None and key.startswith('_raw_'))
				]

		params += [f'{key[5:] if key.startswith("_raw_") else key}={value}'
		           for key, value in self.__dict__.items()
		           if isinstance(value, (int, float, str))
				]

		return f"{self.__class__.__name__}({', '.join(params)})"
