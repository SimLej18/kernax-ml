from __future__ import annotations

import numpy as np
import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..module import AbstractModule
from ..wrappers.WrapperModule import AbstractWrapperModule


def _block_expand(B: Array, sizes1: tuple[int, ...], sizes2: tuple[int, ...]) -> Array:
	"""Repeat ``B[p, q]`` over a block of shape ``(sizes1[p], sizes2[q])``.

	The index arrays are built with ``numpy``, not ``jax.numpy``: the sizes are static,
	so this runs at trace time and produces a gather with a static output shape.
	"""
	i = np.repeat(np.arange(len(sizes1)), sizes1)
	j = np.repeat(np.arange(len(sizes2)), sizes2)
	return B[i[:, None], j[None, :]]


class ICMKernel(AbstractWrapperModule):
	"""Intrinsic Coregionalisation Model: ``K(x1, x2) = B (x) k(x1, x2)``.

	``B = W Wt`` is positive semi-definite by construction, so ``W`` is unconstrained.
	``W`` has shape ``(n_outputs, n_latent)``: ``n_latent = n_outputs`` gives a full-rank
	coregionalisation, ``n_latent < n_outputs`` a low-rank one. ``W`` is deterministically
	initialised to ``eye(n_outputs, n_latent)``; use ``.replace(W=...)`` or
	``kernax.hp_sampling.sample_hps_from_uniform_priors`` to randomise it.

	Two input regimes, selected by ``isotopic_features``:

	- ``True``: ``x1`` is the grid *shared* by every output, shape ``(N, I)``. The result is
	  the dense Kronecker product ``B (x) k(x1, x2)``, shape ``(P*N, P*M)``, in output-major
	  order. ``factors`` exposes the ``(B, K)`` pair so callers can keep the structure.
	- ``False``: ``x1`` is the *concatenation* of the per-output grids, shape
	  ``(sum(feature_sizes), I)``, output-major. The result has shape
	  ``(sum(feature_sizes), sum(feature_sizes2))``, its ``(p, q)`` block being
	  ``B[p, q] * k(x1_p, x2_q)``. An output observed nowhere simply gets size ``0``.

	No own ``engine``: unlike base kernels, this wrapper delegates dispatch entirely to
	``inner`` (same convention as :class:`~kernax.wrappers.ARDKernel.ARDKernel`), so engine
	customisation (e.g. NaN handling, diagonal-only engines) only applies to ``inner``, not
	to the P*N/P*M structure this class builds on top of it.
	"""

	inner: AbstractModule
	W: Array = eqx.field(converter=jnp.asarray)
	isotopic_features: bool = eqx.field(static=True)

	def __init__(self, inner: AbstractModule, n_outputs: int, n_latent: int, isotopic_features: bool = False):
		if n_outputs < 1:
			raise ValueError(f"`n_outputs` must be positive, got {n_outputs}.")
		if n_latent < 1:
			raise ValueError(f"`n_latent` must be positive, got {n_latent}.")
		self.inner = inner
		self.W = jnp.eye(n_outputs, n_latent)
		self.isotopic_features = isotopic_features

	@property
	def n_outputs(self) -> int:
		return self.W.shape[0]

	@property
	def n_latent(self) -> int:
		return self.W.shape[1]

	@property
	def coregionalisation(self) -> Array:
		"""The (P, P) matrix B. Note this property *reduces*: (P, R) -> (P, P)."""
		return self.W @ self.W.T

	def _partition(self, sizes, n: int, name: str) -> tuple[int, ...]:
		if sizes is None:
			raise ValueError(f"`{name}` is required when `isotopic_features` is False.")
		if len(sizes) != self.n_outputs:
			raise ValueError(f"`{name}` must have length P={self.n_outputs}, got {len(sizes)}.")
		if sum(sizes) != n:
			raise ValueError(f"`{name}` must sum to {n}, got {sum(sizes)}.")
		return tuple(sizes)

	def factors(self, x1: Array, x2: Array | None = None, **kwargs) -> tuple[tuple[Array, Array], ...]:
		"""Kronecker terms ``((B, K),)``, following the convention of the sum/product operators."""
		if not self.isotopic_features:
			raise ValueError(
				"`factors` is only defined for isotopic features: the heterotopic block "
				"weighting is not a Kronecker product."
			)
		return ((self.coregionalisation, self.inner(x1, x2, **kwargs)),)

	def __call__(self, x1: Array, x2: Array | None = None, *,
	             feature_sizes: tuple[int, ...] | None = None,
	             feature_sizes2: tuple[int, ...] | None = None,
	             **kwargs) -> Array:
		if self.isotopic_features:
			return sum(jnp.kron(B, K) for B, K in self.factors(x1, x2, **kwargs))
		K = self.inner(x1, x2, **kwargs)
		sizes1 = self._partition(feature_sizes, K.shape[0], "feature_sizes")
		sizes2 = (sizes1 if x2 is None
		          else self._partition(feature_sizes2, K.shape[1], "feature_sizes2"))
		return _block_expand(self.coregionalisation, sizes1, sizes2) * K

	def replace(self, W: None | float | Array = None, **kwargs) -> ICMKernel:
		out = super().replace(**kwargs)
		if W is None:
			return out
		return eqx.tree_at(lambda k: k.W, out, jnp.asarray(W))