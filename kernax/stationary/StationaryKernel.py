import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..AbstractKernel import StaticAbstractKernel
from ..distances import squared_euclidean_distance


class StaticStationaryKernel(StaticAbstractKernel):
	"""
	Super-class for every stationary/isotropic kernel.

	The isotropic property depends only on the distance function used. You can check available
	distance function in `kernax/distances.py`.

	The default distance is the squared Euclidean distance, but it can be overridden in child classes.
	"""

	distance_func = squared_euclidean_distance
