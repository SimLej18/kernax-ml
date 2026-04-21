from __future__ import annotations
from typing import Callable
import equinox as eqx
from jax import Array
from jax import numpy as jnp
from .DotProductKernel import AbstractDotProductKernel
from ..engines import AbstractEngine, DenseEngine
from ..parametrisations import AbstractParametrisation, LogExpParametrisation
from ..distances import dot_product


class SigmoidKernel(AbstractDotProductKernel):
	"""
	Sigmoid (Hyperbolic Tangent) Kernel

	Formula: tanh(α⟨x, x'⟩ + c)

	Parameter alpha must be positive.
	Parameter constant can be any real value.
	"""

	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	_alpha_parametrisation: AbstractParametrisation = eqx.field(static=True)
	_alpha: Array = eqx.field(converter=jnp.asarray)
	constant: Array = eqx.field(converter=jnp.asarray)

	@property
	def alpha(self) -> Array:
		return self._alpha_parametrisation.unwrap(self._alpha)

	def __init__(self,
	             alpha: float | Array,
	             constant: float | Array,
	             alpha_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             distance_function: Callable = dot_product,
	             engine: AbstractEngine = DenseEngine):
		alpha = jnp.asarray(alpha)
		if jnp.any(alpha <= 0):
			raise ValueError("`alpha` must be positive.")

		self.distance_function = distance_function
		self._alpha_parametrisation = alpha_parametrisation
		self._alpha = self._alpha_parametrisation.wrap(alpha)
		self.constant = jnp.asarray(constant)
		self.engine = engine

	def pairwise(self, x1: Array, x2: Array) -> Array:
		dp = self.distance_function(x1, x2)
		return jnp.tanh(self.alpha * dp + self.constant)

	def replace(self,
	            alpha: None | float | Array = None,
	            constant: None | float | Array = None,
	            **kwargs) -> SigmoidKernel:
		new_kernel = self

		if alpha is not None:
			alpha = jnp.asarray(alpha)

			if jnp.any(alpha <= 0):
				raise ValueError("`alpha` must be positive.")

			new_kernel = eqx.tree_at(
				lambda k: k._alpha,
				new_kernel,
				jnp.broadcast_to(
					self._alpha_parametrisation.wrap(alpha),
					self._alpha.shape)
			)

		if constant is not None:
			new_kernel = eqx.tree_at(
				lambda k: k.constant,
				new_kernel,
				jnp.broadcast_to(jnp.asarray(constant), self.constant.shape)
			)

		return new_kernel
