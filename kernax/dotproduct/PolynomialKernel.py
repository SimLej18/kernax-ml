from __future__ import annotations
from typing import Callable
import equinox as eqx
from jax import Array
from jax import numpy as jnp
from .DotProductKernel import AbstractDotProductKernel
from ..engines import AbstractEngine, DenseEngine
from ..parametrisations import AbstractParametrisation, LogExpParametrisation
from ..distances import dot_product


class PolynomialKernel(AbstractDotProductKernel):
	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	degree: int = eqx.field(static=True)
	_gamma_parametrisation: AbstractParametrisation = eqx.field()
	_gamma: Array = eqx.field(converter=jnp.asarray)
	constant: Array = eqx.field(converter=jnp.asarray)

	@property
	def gamma(self) -> Array:
		return self._gamma_parametrisation.unwrap(self._gamma)

	def __init__(self,
	             degree: int,
	             gamma: float | Array,
	             constant: float | Array,
	             gamma_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             distance_function: Callable = dot_product,
	             engine: AbstractEngine = DenseEngine):
		"""
		Initialize the Polynomial kernel.

		Args:
			degree: Degree of the polynomial (must be positive integer)
			gamma: Scale factor (must be positive)
			constant: Independent term (can be any real value)
		"""
		gamma = jnp.asarray(gamma)
		if jnp.any(gamma <= 0):
			raise ValueError("`gamma` must be positive.")

		self.distance_function = distance_function
		self.degree = degree
		self._gamma_parametrisation = gamma_parametrisation
		self._gamma = self._gamma_parametrisation.wrap(gamma)
		self.constant = jnp.asarray(constant)
		self.engine = engine

	def pairwise(self, x1: Array, x2: Array) -> Array:
		return jnp.pow(self.gamma * self.distance_function(x1, x2) + self.constant, self.degree)

	def replace(self,
	            degree: None | int = None,
	            gamma: None | float | Array = None,
	            constant: None | float | Array = None,
	            **kwargs) -> PolynomialKernel:
		new_kernel = self

		if degree is not None:
			raise ValueError("`degree` is a static field and cannot be mutated for PolynomialKernel. "
			                 "Initialise a new kernel instance instead.")

		if gamma is not None:
			gamma = jnp.asarray(gamma)

			if jnp.any(gamma <= 0):
				raise ValueError("`gamma` must be positive.")

			new_kernel = eqx.tree_at(
				lambda k: k._gamma,
				new_kernel,
				jnp.broadcast_to(
					self._gamma_parametrisation.wrap(gamma),
					self._gamma.shape)
			)

		if constant is not None:
			new_kernel = eqx.tree_at(
				lambda k: k.constant,
				new_kernel,
				jnp.broadcast_to(jnp.asarray(constant), self.constant.shape)
			)

		return new_kernel
