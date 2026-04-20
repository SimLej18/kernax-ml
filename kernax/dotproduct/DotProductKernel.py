from __future__ import annotations
from typing import Callable
import equinox as eqx
from jax import Array
from ..AbstractKernel import AbstractKernel


class AbstractDotProductKernel(AbstractKernel):
	"""
	Super-class for every kernel that uses the dot product between input vectors.
	"""
	distance_function: eqx.AbstractVar[Callable[[Array, Array], Array]]