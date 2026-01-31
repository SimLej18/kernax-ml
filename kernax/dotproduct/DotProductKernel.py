import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import StaticAbstractKernel
from ..distances import dot_product


class StaticDotProductKernel(StaticAbstractKernel):
	"""
	Super-class for every kernel that uses the dot product between input vectors.

	This allows to change the distance function used in child classes.
	The default metric is the dot product.
	"""

	distance_func = dot_product
