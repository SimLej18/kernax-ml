"""
Parameter transformation utilities for kernax.

This module provides functions to transform parameters between constrained and
unconstrained spaces, enabling stable optimization of positive-constrained parameters.

The transformation used depends on the global config.parameter_transform setting:
- "identity": No transformation (parameters stay in constrained space)
- "exp": Exponential transform (log-exp trick)
- "softplus": Softplus transform (numerically stable near zero)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def to_unconstrained(value: Array) -> Array:
	"""
	Transform a positive parameter to unconstrained space.

	This function converts a parameter from constrained (positive) space to
	unconstrained (real) space based on the current config.parameter_transform setting.

	Args:
		value: Parameter value in constrained space (must be positive)

	Returns:
		Parameter value in unconstrained space

	Example:
		>>> import kernax
		>>> import jax.numpy as jnp
		>>> kernax.config.parameter_transform = "exp"
		>>> unconstrained = kernax.transforms.to_unconstrained(jnp.array(2.0))
		>>> # unconstrained ≈ 0.693 (log(2.0))
	"""
	from .config import config

	if config.parameter_transform == "identity":
		# No transformation
		return value
	elif config.parameter_transform == "exp":
		# Inverse of exp is log
		return jnp.log(value)
	elif config.parameter_transform == "softplus":
		# Inverse of softplus: log(exp(x) - 1)
		return jnp.log(jnp.exp(value) - 1.0)
	else:
		raise ValueError(f"Unknown parameter_transform: {config.parameter_transform}")


def to_constrained(value: Array) -> Array:
	"""
	Transform a parameter from unconstrained space to positive constrained space.

	This function converts a parameter from unconstrained (real) space back to
	constrained (positive) space based on the current config.parameter_transform setting.

	Args:
		value: Parameter value in unconstrained space

	Returns:
		Parameter value in constrained space (always positive)

	Example:
		>>> import kernax
		>>> import jax.numpy as jnp
		>>> kernax.config.parameter_transform = "exp"
		>>> constrained = kernax.transforms.to_constrained(jnp.array(0.693))
		>>> # constrained ≈ 2.0 (exp(0.693))
	"""
	from .config import config

	if config.parameter_transform == "identity":
		# No transformation
		return value
	elif config.parameter_transform == "exp":
		# Exponential ensures positivity
		return jnp.exp(value)
	elif config.parameter_transform == "softplus":
		# Softplus: log(1 + exp(x))
		return jnp.log(1.0 + jnp.exp(value))
	else:
		raise ValueError(f"Unknown parameter_transform: {config.parameter_transform}")
