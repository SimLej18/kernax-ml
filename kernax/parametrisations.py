"""
Parameter transformation utilities for kernax.

This module provides modules to transform parameters between constrained and
unconstrained spaces, enabling stable optimization of positive-constrained parameters.
"""

from __future__ import annotations
from abc import abstractmethod
import jax.numpy as jnp
from jax import Array, jit
import equinox as eqx


class AbstractParametrisation(eqx.Module):
	@classmethod
	@abstractmethod
	def wrap(cls, param: Array) -> Array:
		raise NotImplementedError

	@classmethod
	@abstractmethod
	def unwrap(cls, param: Array) -> Array:
		raise NotImplementedError


class LogExpParametrisation(AbstractParametrisation):
	@staticmethod
	@jit
	def wrap(param: Array) -> Array:
		return jnp.log(param)  # From R+ to R

	@staticmethod
	@jit
	def unwrap(param: Array) -> Array:
		return jnp.exp(param)  # From R to R+


# TODO: 1-logexp constraint
# TODO: sigmoid constraint for border optimisation
