from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional, Type

import equinox as eqx
import jax.numpy as jnp
from equinox import filter_jit
from jax import Array, vmap
from jax.lax import cond

from .engines import ComputationEngine, DenseEngine
from .utils import format_jax_array

if TYPE_CHECKING:
	pass


class AbstractKernel(eqx.Module):
	static_class: ClassVar[Optional[Type[StaticAbstractKernel]]] = None
	computation_engine: Type[ComputationEngine] = eqx.field(
		static=True, default=DenseEngine, kw_only=True
	)

	def __init__(self, computation_engine=DenseEngine, **kwargs):
		"""
		Initialize the kernel and mark that a kernel has been instantiated.

		This locks the parameter_transform config setting to prevent inconsistencies
		with JIT-compiled code.

		Args:
			computation_engine: The computation engine to use for covariance calculations
			**kwargs: Additional keyword arguments (for subclass compatibility)
		"""
		# Import here to avoid circular dependency
		from .config import config

		# Mark that kernels have been instantiated (locks parameter_transform)
		config._mark_kernel_instantiated()

		# Set the computation engine
		self.computation_engine = computation_engine

	def replace(self, **kwargs):
		"""API de modification fonctionnelle (Setter idiomatique)."""
		from .transforms import to_unconstrained

		# Adapter les paramètres contraints et broadcaster si nécessaire
		adapted = {}
		for k, v in kwargs.items():
			raw_field = f"_raw_{k}"

			if hasattr(self, raw_field):
				# Paramètre contraint : valider, broadcaster, transformer
				v = jnp.asarray(v)
				v = eqx.error_if(v, jnp.any(v <= 0), f"{k} must be positive.")

				# Broadcaster à la shape actuelle si nécessaire
				current = getattr(self, raw_field)
				if current.shape != v.shape:
					v = jnp.broadcast_to(v, current.shape)

				adapted[raw_field] = to_unconstrained(v)
			else:
				# Paramètre non-contraint : broadcaster si nécessaire pour les Arrays
				if hasattr(self, k):
					current = getattr(self, k)

					# Broadcaster uniquement si current et v sont des Arrays
					if isinstance(current, Array):
						v = jnp.asarray(v) if not isinstance(v, Array) else v

						# Broadcaster si shapes différentes
						if current.shape != v.shape:
							v = jnp.broadcast_to(v, current.shape)

				adapted[k] = v

		where = lambda s: [getattr(s, k) for k in adapted.keys()]
		return eqx.tree_at(where, self, list(adapted.values()))

	@filter_jit
	def __call__(self, x1: Array, x2: Optional[Array] = None) -> Array:
		# If no x2 is provided, we compute the covariance between x1 and itself
		if x2 is None:
			x2 = x1

		# Turn scalar inputs into vectors
		x1, x2 = jnp.atleast_1d(x1), jnp.atleast_1d(x2)

		# Ensure static_class is not None
		assert self.static_class is not None, "static_class must be defined in subclass"

		# Call the appropriate method
		if jnp.ndim(x1) == 1 and jnp.ndim(x2) == 1:
			return self.computation_engine.pairwise_cov_if_not_nan(self, x1, x2)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 1:
			return self.computation_engine.cross_cov_vector_if_not_nan(self, x1, x2)
		elif jnp.ndim(x1) == 1 and jnp.ndim(x2) == 2:
			return self.computation_engine.cross_cov_vector_if_not_nan(self, x2, x1)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 2:
			return self.computation_engine.cross_cov_matrix(self, x1, x2)
		else:
			raise ValueError(
				f"Invalid input dimensions: x1 has shape {x1.shape}, x2 has shape {x2.shape}. "
				"Expected scalar, 1D or 2D arrays as inputs."
			)

	def __add__(self, other):
		from kernax.operators import SumKernel

		return SumKernel(self, other)

	def __radd__(self, other):
		from kernax.operators import SumKernel

		return SumKernel(other, self)

	def __sub__(self, other):
		from kernax.operators import SumKernel
		from kernax.wrappers import NegKernel

		return SumKernel(self, NegKernel(other))

	def __rsub__(self, other):
		from kernax.operators import SumKernel
		from kernax.wrappers import NegKernel

		return SumKernel(other, NegKernel(self))

	def __neg__(self):
		from kernax.wrappers import NegKernel

		return NegKernel(self)

	def __mul__(self, other):
		from kernax.operators import ProductKernel

		return ProductKernel(self, other)

	def __rmul__(self, other):
		from kernax.operators import ProductKernel

		return ProductKernel(other, self)

	def __str__(self):
		from kernax.transforms import to_constrained

		# Print parameters, aka elements of __dict__ that are jax arrays
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


class StaticAbstractKernel:
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return jnp.array(jnp.nan)  # To be overwritten in subclasses

	@classmethod
	@filter_jit
	def pairwise_cov_if_not_nan(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Returns NaN if either x1 or x2 is NaN, otherwise calls the pairwise_cov method.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return cond(  # type: ignore[no-any-return]
			jnp.any(jnp.isnan(x1) | jnp.isnan(x2)),
			lambda _: jnp.nan,
			lambda _: cls.pairwise_cov(kern, x1, x2),
			None,
		)

	@classmethod
	@filter_jit
	def cross_cov_vector(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel cross covariance values between an array of vectors (matrix) and a vector.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: scalar array
		:return: vector array (N, )
		"""
		return vmap(lambda x: cls.pairwise_cov_if_not_nan(kern, x, x2), in_axes=0)(x1)

	@classmethod
	@filter_jit
	def cross_cov_vector_if_not_nan(
		cls, kern: AbstractKernel, x1: Array, x2: Array, **kwargs
	) -> Array:
		"""
		Returns an array of NaN if scalar is NaN, otherwise calls the cross_cov_vector method.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return cond(  # type: ignore[no-any-return]
			jnp.any(jnp.isnan(x2)),
			lambda _: jnp.full(len(x1), jnp.nan),
			lambda _: cls.cross_cov_vector(kern, x1, x2),
			None,
		)

	@classmethod
	@filter_jit
	def cross_cov_matrix(cls, kern: AbstractKernel, x1: Array, x2: Array) -> Array:
		"""
		Compute the kernel covariance matrix between two vector arrays.

		:param kern: kernel instance containing the hyperparameters
		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:return: matrix array (N, M)
		"""
		return vmap(lambda x: cls.cross_cov_vector_if_not_nan(kern, x2, x), in_axes=0)(x1)
