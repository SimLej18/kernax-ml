from __future__ import annotations

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..parametrisations import AbstractParametrisation, LogExpParametrisation
from ..types import KernelLike
from ..wrappers.WrapperModule import AbstractWrapperModule


def _check_kappa(kappa: float | Array, n_outputs: int) -> Array:
	kappa = jnp.broadcast_to(jnp.asarray(kappa, dtype=float), (n_outputs,))
	if jnp.any(kappa <= 0):
		raise ValueError("`kappa` must be strictly positive.")
	return kappa


class ICMKernel(AbstractWrapperModule[KernelLike]):
	"""Intrinsic Coregionalisation Model: ``K(x1, x2) = B (x) k(x1, x2)``.

	``B = W Wt + diag(kappa)`` is positive semi-definite by construction, so ``W`` is
	unconstrained. ``W`` has shape ``(n_outputs, n_latent)``: ``n_latent = n_outputs`` gives
	a full-rank ``W Wt``, ``n_latent < n_outputs`` a low-rank one. ``kappa`` is a ``(P,)``
	vector of strictly positive per-output variances, which makes ``B`` positive *definite*
	even at low rank. Both are deterministically initialised (``W = eye(n_outputs,
	n_latent)``, ``kappa = 1``); use ``.replace(W=..., kappa=...)`` or
	``kernax.hp_sampling.sample_hps_from_uniform_priors`` to randomise them.

	``kappa`` scales the whole within-output block of ``K``, not just its diagonal: it is
	output-specific *signal* variance, not observation noise -- the latter is a
	:class:`~kernax.multioutput.BlockDiagKernel.BlockDiagKernel` over a ``WhiteNoiseKernel``,
	added alongside. Pass ``kappa_parametrisation=NonTrainableParametrisation()`` to hold it
	fixed during optimisation.

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

	inner: KernelLike
	W: Array = eqx.field(converter=jnp.asarray)
	_kappa_parametrisation: AbstractParametrisation = eqx.field()
	_kappa: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, inner: KernelLike, n_outputs: int, n_latent: int,
	             kappa: float | Array = 1.0,
	             kappa_parametrisation: AbstractParametrisation = LogExpParametrisation()):
		if n_outputs < 1:
			raise ValueError(f"`n_outputs` must be positive, got {n_outputs}.")
		if n_latent < 1:
			raise ValueError(f"`n_latent` must be positive, got {n_latent}.")
		self.inner = inner
		self.W = jnp.eye(n_outputs, n_latent)
		self._kappa_parametrisation = kappa_parametrisation
		self._kappa = kappa_parametrisation.wrap(_check_kappa(kappa, n_outputs))

	@property
	def n_outputs(self) -> int:
		return self.W.shape[0]

	@property
	def n_latent(self) -> int:
		return self.W.shape[1]

	@property
	def kappa(self) -> Array:
		return self._kappa_parametrisation.unwrap(self._kappa)

	@property
	def coregionalisation(self) -> Array:
		"""The (P, P) matrix B = W Wt + diag(kappa). Note this property *reduces*: (P, R) -> (P, P)."""
		return self.W @ self.W.T + jnp.diag(self.kappa)

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

	def spectral_density(self, w: Array) -> Array:
		raise NotImplementedError(
			"`ICMKernel` has no scalar spectral density: coregionalisation makes it the "
			"matrix-valued `B * S(w)`, which this class does not expose. Use "
			"`self.coregionalisation` and `self.inner.spectral_density(w)` to build it."
		)

	def replace(self, W: None | float | Array = None, kappa: None | float | Array = None,
	            **kwargs) -> ICMKernel:
		out = super().replace(**kwargs)
		if W is not None:
			out = eqx.tree_at(lambda k: k.W, out, jnp.asarray(W))
		if kappa is not None:
			out = eqx.tree_at(
				lambda k: k._kappa,
				out,
				out._kappa_parametrisation.wrap(_check_kappa(kappa, out.n_outputs))
			)
		return out
