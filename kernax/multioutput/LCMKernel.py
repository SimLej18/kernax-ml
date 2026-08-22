from __future__ import annotations

from typing import Sequence

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..module import AbstractModule
from ..types import KernelLike
from .ICMKernel import ICMKernel


class LCMKernel(AbstractModule):
	"""Linear Coregionalisation Model: ``K(x1, x2) = sum_q B_q (x) k_q(x1, x2)``.

	``Q`` independent kernels, each paired with its own coregionalisation matrix
	``B_q = W_q W_qt + diag(kappa_q)`` -- arbitrary rank and arbitrary kernel
	class/hyperparameters per component. Wraps each ``(kernel_q, W_q, kappa_q)`` triple into
	an :class:`~kernax.multioutput.ICMKernel.ICMKernel` and sums their outputs in a plain
	Python loop over the ``Q`` components
	"""

	components: tuple[ICMKernel, ...]

	def __init__(self, kernels: Sequence[KernelLike],
	             coregionalization_matrices: Sequence[float | Array],
	             kappas: Sequence[float | Array] | None = None):
		if len(kernels) < 1:
			raise ValueError("`kernels` must contain at least one kernel.")
		if len(kernels) != len(coregionalization_matrices):
			raise ValueError(
				f"`kernels` and `coregionalization_matrices` must have the same length, "
				f"got {len(kernels)} and {len(coregionalization_matrices)}."
			)
		if kappas is None:
			kappas = [1.0] * len(kernels)
		elif len(kappas) != len(kernels):
			raise ValueError(
				f"`kappas` and `kernels` must have the same length, "
				f"got {len(kappas)} and {len(kernels)}."
			)

		components = []
		for kernel, W, kappa in zip(kernels, coregionalization_matrices, kappas, strict=True):
			W = jnp.asarray(W)
			if W.ndim != 2:
				raise ValueError("each coregionalisation matrix must be a (P, R) matrix.")
			icm = ICMKernel(kernel, W.shape[0], W.shape[1], kappa=kappa)
			components.append(icm.replace(W=W))

		n_outputs = components[0].n_outputs
		if any(c.n_outputs != n_outputs for c in components):
			raise ValueError("all coregionalisation matrices must share the same number of outputs P.")

		self.components = tuple(components)

	@property
	def n_components(self) -> int:
		return len(self.components)

	def __call__(self, x1: Array, x2: Array | None = None, **kwargs) -> Array:
		return sum(c(x1, x2, **kwargs) for c in self.components)

	def factors(self, x1: Array, x2: Array | None = None, **kwargs) -> tuple[tuple[Array, Array], ...]:
		"""Kronecker terms ``((B_1, K_1), ...)``, following the sum/product operator convention.

		Only meaningful for the shared-grid case: call without ``output_ids``.
		"""
		return tuple(c.factors(x1, x2, **kwargs)[0] for c in self.components)

	def spectral_density(self, w: Array) -> Array:
		raise NotImplementedError(
			"`LCMKernel` has no scalar spectral density: it is the matrix-valued "
			"`sum_q B_q * S_q(w)`, which this class does not expose. Build it from "
			"`self.components`."
		)

	def replace(self, **kwargs) -> LCMKernel:
		"""Broadcast ``kwargs`` to every component via ``ICMKernel.replace``."""
		return eqx.tree_at(
			lambda m: m.components,
			self,
			tuple(c.replace(**kwargs) for c in self.components)
		)
