from typing import Callable
import equinox as eqx
from jax import Array
from ..AbstractKernel import AbstractKernel


class AbstractDotProductKernel(AbstractKernel):
	"""
	Super-class for every kernel that uses the dot product between input vectors.

	This allows to change the distance function used in child classes.
	"""
	distance_func = eqx.AbstractVar[Callable[[Array, Array], Array]]
