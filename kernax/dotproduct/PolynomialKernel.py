from __future__ import annotations
import equinox as eqx
from equinox import filter_jit
from jax import Array
from jax import numpy as jnp
from .DotProductKernel import AbstractDotProductKernel
from ...engines import AbstractEngine, DenseEngine
from ...parametrisations import AbstractParametrisation, LogExpParametrisation
from ...distances import dot_product


class PolynomialKernel(AbstractDotProductKernel):
	engine: AbstractEngine = eqx.field(static=True)
	_distance_function: Callable = eqx.field(static=True)
	degree: int = eqx.field(static=True)
	_gamma_parametrisation: AbstractParametrisation = eqx.field(static=True)
	_gamma: Array = eqx.field(converter=jnp.asarray)
	constant: Array = eqx.field(converter=jnp.asarray)

	@property
	def gamma(self) -> Array:
		return self._gamma_parametrisation.unwrap(self._gamma)

	def __init__(self,
	             degree: int,
	             gamma: float|Array,
	             constant: float|Array,
	             gamma_parametrisation: AbstractParametrisation = LogExpParametrisation,
	             distance_function: Callable = dot_product,
	             engine: AbstractEngine = DenseEngine):
		"""
		Initialize the Polynomial kernel.

		Args:
			degree: Degree of the polynomial (must be positive integer)
			gamma: Scale factor (must be positive)
			constant: Independent term (can be any real value)

		Raises:
			ValueError: If degree is not positive or gamma is not positive
		"""
		# Assert gamma is positive
		assert jnp.all(jnp.asarray(gamma) > 0).item(),  "`gamma` must be non-negative."

		self._distance_function = distance_function
		self.degree = degree
		self._gamma_parametrisation = gamma_parametrisation
		self._gamma = self._gamma_parametrisation.wrap(gamma)
		self.constant = constant
		self.engine = engine


	@filter_jit
	def pairwise(self, x1: Array, x2: Array) -> Array:
		"""
		Compute the polynomial kernel covariance value between two vectors.

		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return jnp.pow(self.gamma * self._distance_function(x1, x2) + self.constant, self.degree)

	def replace(self,
	            degree: None | int = None,
	            gamma: None | float | Array = None,
	            constant: None | float | Array = None,
	            **kwargs) -> PolynomialKernel:
		new_kernel = self

		if degree is not None:
			new_kernel = eqx.tree_at(
				lambda k: k.degree,
				new_kernel,
				degree
			)

		if gamma is not None:
			new_kernel = eqx.tree_at(
				lambda k: k.gamma,
				new_kernel,
				jnp.broadcast_to(
					self._gamma_parametrisation.wrap(jnp.asarray(gamma)),
					self.gamma.shape),
			)

		if constant is not None:
			new_kernel = eqx.tree_at(
				lambda k: k.constant,
				new_kernel,
				jnp.broadcast_to(
					jnp.asarray(constant),
					self.constant.shape),
			)

		return new_kernel
