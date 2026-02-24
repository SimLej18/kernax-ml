"""
Classes that extend Equinox' Module to provide state-structure separation.

A module is either a Static class or an instance. The instance is linked to a static class
to make computations.

These classes are mainly here to be extended by Kernel and Mean classes, while providing
type-checking for functions that receive a Module as an argument.
"""

from typing import TYPE_CHECKING, ClassVar, Optional, Type
import equinox as eqx
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
	pass


class StaticAbstractModule:
	pass  # Nothing special here


class AbstractModule(eqx.Module):
	static_class: ClassVar[Optional[Type[StaticAbstractModule]]] = None

	def __init__(self, **kwargs):
		"""
		Initialize the module and mark that a module has been instantiated.

		This locks the parameter_transform config setting to prevent inconsistencies
		with JIT-compiled code.

		Args:
			**kwargs: Additional keyword arguments (for subclass compatibility)
		"""
		super().__init__(**kwargs)

		# Import here to avoid circular dependency
		from .config import config

		# Mark that kernels have been instantiated (locks parameter_transform)
		config._mark_module_instantiated()

	def __call__(self, *args, **kwargs):
		raise NotImplementedError("Subclasses must implement __call__ method.")

	def replace(self, **kwargs):
		"""API de modification fonctionnelle (Setter idiomatique)."""
		from .transforms import to_unconstrained

		adapted = {}
		for k, v in kwargs.items():
			raw_field = f"_raw_{k}"

			if hasattr(self, raw_field):
				v = jnp.asarray(v)
				v = eqx.error_if(v, jnp.any(v <= 0), f"{k} must be positive.")

				current = getattr(self, raw_field)
				if current.shape != v.shape:
					v = jnp.broadcast_to(v, current.shape)

				adapted[raw_field] = to_unconstrained(v)
			elif hasattr(self, k):
				current = getattr(self, k)

				if isinstance(current, Array):
					v = jnp.asarray(v) if not isinstance(v, Array) else v

					if current.shape != v.shape:
						v = jnp.broadcast_to(v, current.shape)

				adapted[k] = v

		where = lambda s: [getattr(s, k) for k in adapted.keys()]
		return eqx.tree_at(where, self, list(adapted.values()))

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
		from kernax.transforms import to_constrained
		from kernax.utils import format_jax_array

		return f"{self.__class__.__name__}({
			', '.join(
				[
					f'{key}={format_jax_array(value)}' if '_raw_' not in key else f'{key[5:]}={format_jax_array(to_constrained(value))}'
					for key, value in self.__dict__.items()
					if isinstance(value, Array)
				]
				+
				[
					f'{key}={value}' for key, value in self.__dict__.items() if isinstance(value, (int, float, str))
				]
			)
		})"