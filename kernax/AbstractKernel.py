from __future__ import annotations
from abc import abstractmethod
import equinox as eqx
from jax import Array
from .module import AbstractModule
from .engines import AbstractEngine


class AbstractKernel(AbstractModule):
	engine: eqx.AbstractVar[AbstractEngine]

	@abstractmethod
	def pairwise(self, x1: Array, x2: Array):
		raise NotImplementedError

	def __call__(self, x1: Array, x2: Array | None = None, *args, **kwargs) -> Array:
		if x2 is None:
			x2 = x1

		return self.engine.__call__(self, x1, x2, *args, **kwargs)

	@abstractmethod
	def replace(self, x: Array) -> Array:
		raise NotImplementedError
