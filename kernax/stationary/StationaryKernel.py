from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import equinox as eqx
from jax import Array

from ..AbstractKernel import AbstractKernel


class AbstractStationaryKernel(AbstractKernel):
	"""
	Super-class for every stationary/isotropic kernel.

	The isotropic property depends only on the distance function used. You can check available
	distance function in `kernax/distances.py`.
	"""
	distance_function: eqx.AbstractVar[Callable[[Array, Array], Array]]

	@abstractmethod
	def spectral_density(self, w: Array) -> Array:
		raise NotImplementedError
