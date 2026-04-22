"""
Parameter transformation utilities for kernax.

This module provides modules to transform parameters between constrained and
unconstrained spaces, enabling stable optimization of positive-constrained parameters.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Iterable
import jax.nn
import jax.numpy as jnp
import jax.lax as jlx
from jax import Array
import equinox as eqx


class AbstractParametrisation(eqx.Module):
	@abstractmethod
	def wrap(self, param: Array) -> Array:
		raise NotImplementedError

	@abstractmethod
	def unwrap(self, param: Array) -> Array:
		raise NotImplementedError


class ParametrisationChain(AbstractParametrisation):
	parametrisations: Iterable[AbstractParametrisation]

	def wrap(self, param: Array) -> Array:
		for parametrisation in self.parametrisations:
			param = parametrisation.wrap(param)
		return param

	def unwrap(self, param: Array) -> Array:
		for parametrisation in reversed(self.parametrisations):
			param = parametrisation.unwrap(param)
		return param


class IdentityParametrisation(AbstractParametrisation):
	def wrap(self, param: Array) -> Array:
		return param

	def unwrap(self, param: Array) -> Array:
		return param


class NonTrainableParametrisation(AbstractParametrisation):
	def wrap(self, param: Array) -> Array:
		return param

	def unwrap(self, param: Array) -> Array:
		return jlx.stop_gradient(param)


class LogExpParametrisation(AbstractParametrisation):
	def wrap(self, param: Array) -> Array:
		return jnp.log(param)  # From R+ to R

	def unwrap(self, param: Array) -> Array:
		return jnp.exp(param)  # From R to R+


class SoftPlusParametrisation(AbstractParametrisation):
	def wrap(self, param: Array) -> Array:
		return jnp.log(jnp.exp(param) - 1)  # From R+ to R

	def unwrap(self, param: Array) -> Array:
		return jnp.log(1 + jnp.exp(param))  # From R to R+


class BoundedParametrisation(AbstractParametrisation):
	lower_bound: Array = eqx.field(converter=jnp.asarray)
	upper_bound: Array = eqx.field(converter=jnp.asarray)

	def wrap(self, param: Array) -> Array:
		# From (lower_bound, upper_bound) to R
		return jax.scipy.special.logit((param - self.lower_bound) / (self.upper_bound - self.lower_bound))

	def unwrap(self, param: Array) -> Array:
		# From R to (lower_bound, upper_bound)
		return self.lower_bound + (self.upper_bound - self.lower_bound) * jax.nn.sigmoid(param)
