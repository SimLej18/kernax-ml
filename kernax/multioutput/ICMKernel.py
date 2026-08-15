from __future__ import annotations

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..module import AbstractModule
from ..wrappers.WrapperModule import AbstractWrapperModule


class ICMKernel(AbstractWrapperModule):
	"""Intrinsic Coregionalisation Model: ``K(x1, x2) = B (x) k(x1, x2)``.

	``B = W Wt`` is positive semi-definite by construction, so ``W`` is unconstrained.
	``W`` has shape ``(n_outputs, n_latent)``: ``n_latent = n_outputs`` gives a full-rank
	coregionalisation, ``n_latent < n_outputs`` a low-rank one. ``W`` is deterministically
	initialised to ``eye(n_outputs, n_latent)``; use ``.replace(W=...)`` or
	``kernax.hp_sampling.sample_hps_from_uniform_priors`` to randomise it.

	Two input regimes, selected by whether ``output_ids`` is passed to ``__call__`` --
	this is a property of the data, not of the kernel, so it is not stored on the instance:

	- Omitted: ``x1`` is the grid *shared* by every output, shape ``(N, I)``. The result is
	  the dense Kronecker product ``B (x) k(x1, x2)``, shape ``(P*N, P*M)``, in output-major
	  order. ``factors`` exposes the ``(B, K)`` pair so callers can keep the structure.
	- Given: ``x1`` holds points from every output, in *any* order, shape ``(N, I)``;
	  ``output_ids`` (and ``output_ids2`` for cross-covariances) is an integer array of
	  shape ``(N,)`` giving, for each row, which of the ``P`` outputs it belongs to. The
	  result has shape ``(N, M)``, entry ``(i, j)`` being
	  ``B[output_ids[i], output_ids2[j]] * k(x1[i], x2[j])``. Grid-aligned (output-major)
	  data still works: ``output_ids = jnp.repeat(jnp.arange(P), feature_sizes)``.

	No own ``engine``: unlike base kernels, this wrapper delegates dispatch entirely to
	``inner`` (same convention as :class:`~kernax.wrappers.ARDKernel.ARDKernel`), so engine
	customisation (e.g. NaN handling, diagonal-only engines) only applies to ``inner``, not
	to the N/M structure this class builds on top of it.
	"""

	inner: AbstractModule
	W: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, inner: AbstractModule, n_outputs: int, n_latent: int):
		if n_outputs < 1:
			raise ValueError(f"`n_outputs` must be positive, got {n_outputs}.")
		if n_latent < 1:
			raise ValueError(f"`n_latent` must be positive, got {n_latent}.")
		self.inner = inner
		self.W = jnp.eye(n_outputs, n_latent)

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

	def factors(self, x1: Array, x2: Array | None = None, **kwargs) -> tuple[tuple[Array, Array], ...]:
		"""Kronecker terms ``((B, K),)``, following the convention of the sum/product operators.

		Only meaningful for the shared-grid case: call without ``output_ids``.
		"""
		return ((self.coregionalisation, self.inner(x1, x2, **kwargs)),)

	def __call__(self, x1: Array, x2: Array | None = None, *,
	             output_ids: Array | None = None,
	             output_ids2: Array | None = None,
	             **kwargs) -> Array:
		if output_ids is None:
			if output_ids2 is not None:
				raise ValueError("`output_ids2` requires `output_ids`.")
			return sum(jnp.kron(B, K) for B, K in self.factors(x1, x2, **kwargs))

		output_ids = jnp.asarray(output_ids)
		if output_ids.shape[0] != x1.shape[0]:
			raise ValueError(
				f"`output_ids` must have length {x1.shape[0]} (= len(x1)), got {output_ids.shape[0]}.")
		if x2 is None:
			output_ids2 = output_ids
		elif output_ids2 is None:
			raise ValueError("`output_ids2` is required when `x2` is given.")
		else:
			output_ids2 = jnp.asarray(output_ids2)
			if output_ids2.shape[0] != x2.shape[0]:
				raise ValueError(
					f"`output_ids2` must have length {x2.shape[0]} (= len(x2)), got {output_ids2.shape[0]}.")

		K = self.inner(x1, x2, **kwargs)
		B = self.coregionalisation
		return B[output_ids[:, None], output_ids2[None, :]] * K

	def replace(self, W: None | float | Array = None, **kwargs) -> ICMKernel:
		out = super().replace(**kwargs)
		if W is None:
			return out
		return eqx.tree_at(lambda k: k.W, out, jnp.asarray(W))
