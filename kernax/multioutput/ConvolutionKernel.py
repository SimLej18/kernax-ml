from __future__ import annotations

from typing import Callable

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp

from ..AbstractKernel import AbstractKernel
from ..distances import squared_euclidean_distance
from ..engines import AbstractEngine
from ..parametrisations import AbstractParametrisation, LogExpParametrisation


class _ConvolutionEngine(AbstractEngine):
	"""Vmaps ``pairwise`` over both point axes, carrying each row's output index alongside its
	coordinates so the two axes can hold points belonging to different outputs.

	Not part of the swappable ``engines`` catalogue: :meth:`ConvolutionKernel.pairwise` takes
	two extra positional arguments (the output indices), so it is not compatible with
	``DenseEngine`` and friends. This is the only engine :class:`ConvolutionKernel` can use.
	"""
	@staticmethod
	def __call__(module: ConvolutionKernel, x1: Array, x2: Array, idx1: Array, idx2: Array) -> Array:
		gram = vmap(vmap(module.pairwise, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))
		return gram(x1, x2, idx1, idx2)


class ConvolutionKernel(AbstractKernel):
	r"""Multi-output convolution process kernel (Alvarez & Lawrence, 2011).

	Every output is the same latent white-noise field seen through its own Gaussian
	smoothing window, so the covariance between outputs ``o`` and ``o'`` is an RBF whose
	bandwidth is the *sum* of the two windows:

	.. math::
	    [k(x, x')]_{o,o'} = \sigma_o\,\sigma_{o'}\,\rho_{o,o'}
	        \exp\left(-\frac12 \sum_i \frac{(x_i - x'_i)^2}{a_{o,i} + a_{o',i}}\right),
	    \qquad
	    \rho_{o,o'} = \prod_i \left(\frac{2\sqrt{a_{o,i}\,a_{o',i}}}{a_{o,i} + a_{o',i}}\right)^{1/2}

	``variance`` holds :math:`\sigma_o^2`, the marginal variance of output ``o``, and
	``bandwidth`` holds :math:`a_{o,i}`, giving output ``o`` a marginal length-scale of
	:math:`\sqrt{2 a_{o,i}}` along axis ``i``.

	Positive semi-definiteness holds for any positive ``bandwidth``. Note that
	:math:`\rho_{o,o'} \le 1`, with equality iff ``a[o] == a[o']``: two outputs of different
	bandwidth cannot be strongly correlated under this model. That is structural to the
	convolution-process construction, not a fitting artefact.

	By default (``ard=False``) each output shares a single bandwidth across every input
	dimension, so ``bandwidth`` has shape ``(n_outputs,)``. Pass ``ard=True`` to give every
	output its own bandwidth per input dimension instead, in which case ``bandwidth`` must
	have shape ``(n_outputs, n_features)``.

	Two input regimes, selected by whether ``output_ids`` is passed to ``__call__`` -- this
	is a property of the data, not of the kernel, so it is not stored on the instance:

	- Omitted: ``x1`` is the grid *shared* by every output, shape ``(N, I)``; ``x2``, if
	    given, is another such shared grid, shape ``(M, I)``. The result has shape
	    ``(P*N, P*M)``, output-major, equivalent to tiling ``x1``/``x2`` across every output
	    and calling with ``output_ids`` explicitly. Unlike
	    :class:`~kernax.multioutput.ICMKernel.ICMKernel`, this is a convenience, not a
	    speed-up: cross-output correlation here is derived from per-output bandwidth, not a
	    free Kronecker factor, so every ``(o, o')`` block still has to be computed
	    individually.
	- Given: ``x1`` holds points from every output, in any order, shape ``(N, I)``;
	    ``output_ids`` (and, for cross-covariances between two different sets of points,
	    ``output_ids2``) is an integer array of shape ``(N,)`` giving, for each row of
	    ``x1``, which of the ``n_outputs`` outputs it belongs to.

	:param variance: marginal variance of each output, positive, shape ``(n_outputs,)``
	:param bandwidth: bandwidth of each output, positive; shape ``(n_outputs,)`` if
	    ``ard=False`` (default), ``(n_outputs, n_features)`` if ``ard=True``
	:param ard: whether every input dimension gets its own bandwidth per output
	:param distance_function: squared-distance function applied to the bandwidth-rescaled
	    points; defaults to the Euclidean squared distance, see ``kernax.distances``

	There is no ``engine`` parameter: unlike other kernels, ``pairwise`` here takes an extra
	output index per point, which the shared engines (``DenseEngine`` and friends) don't
	pass through. Computation always goes through this kernel's own engine.

	Use :meth:`from_paper_parameters` to build a kernel directly from the parametrisation of
	Alvarez & Lawrence (2011); this class uses an equivalent but better-behaved
	parametrisation that decouples amplitude from bandwidth and removes a flat likelihood
	direction present in the paper's own parameters.
	"""

	engine: AbstractEngine = eqx.field(static=True)
	distance_function: Callable = eqx.field(static=True)
	ard: bool = eqx.field(static=True)
	_variance_parametrisation: AbstractParametrisation = eqx.field()
	_variance: Array = eqx.field(converter=jnp.asarray)
	_bandwidth_parametrisation: AbstractParametrisation = eqx.field()
	_bandwidth: Array = eqx.field(converter=jnp.asarray)

	def __init__(self,
	             variance: float | Array,
	             bandwidth: float | Array,
	             ard: bool = False,
	             variance_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             bandwidth_parametrisation: AbstractParametrisation = LogExpParametrisation(),
	             distance_function: Callable = squared_euclidean_distance):
		variance = jnp.atleast_1d(jnp.asarray(variance))
		bandwidth = jnp.asarray(bandwidth)

		expected_ndim = 2 if ard else 1
		if bandwidth.ndim != expected_ndim:
			shape_hint = "(n_outputs, n_features)" if ard else "(n_outputs,)"
			raise ValueError(
				f"`bandwidth` must have shape {shape_hint} when ard={ard}, got {bandwidth.shape}.")
		if variance.shape != bandwidth.shape[:1]:
			raise ValueError(
				f"`variance` must have shape (n_outputs,) = {bandwidth.shape[:1]}, got {variance.shape}.")
		if jnp.any(variance <= 0):
			raise ValueError("`variance` must be positive.")
		if jnp.any(bandwidth <= 0):
			raise ValueError("`bandwidth` must be positive.")

		self.ard = ard
		self.distance_function = distance_function
		self._variance_parametrisation = variance_parametrisation
		self._variance = variance_parametrisation.wrap(variance)
		self._bandwidth_parametrisation = bandwidth_parametrisation
		self._bandwidth = bandwidth_parametrisation.wrap(bandwidth)
		self.engine = _ConvolutionEngine

	@property
	def variance(self) -> Array:
		return self._variance_parametrisation.unwrap(self._variance)

	@property
	def bandwidth(self) -> Array:
		return self._bandwidth_parametrisation.unwrap(self._bandwidth)

	@property
	def n_outputs(self) -> int:
		return self._bandwidth.shape[0]

	def pairwise(self, x1: Array, x2: Array, feature1: Array, feature2: Array) -> Array:
		"""Covariance between point ``x1`` of output ``feature1`` and ``x2`` of output ``feature2``."""
		a = self.bandwidth
		sigma = jnp.sqrt(self.variance)
		a1, a2 = a[feature1], a[feature2]
		s = a1 + a2

		# `broadcast_to(..., x1.shape)` makes the log-determinant terms correct whether `a`
		# is per-output (ard=False, one value broadcast over all `n_features` dimensions) or
		# per-output-per-dimension (ard=True, already the right shape).
		log_det_2a1 = jnp.sum(jnp.broadcast_to(jnp.log(2.0 * a1), x1.shape))
		log_det_2a2 = jnp.sum(jnp.broadcast_to(jnp.log(2.0 * a2), x1.shape))
		log_det_s = jnp.sum(jnp.broadcast_to(jnp.log(s), x1.shape))
		log_rho = 0.25 * (log_det_2a1 + log_det_2a2) - 0.5 * log_det_s

		scale = jnp.sqrt(s)
		quad = self.distance_function(x1 / scale, x2 / scale)
		return sigma[feature1] * sigma[feature2] * jnp.exp(log_rho - 0.5 * quad)

	def __call__(self, x1: Array, x2: Array | None = None, *,
	             output_ids: Array | None = None,
	             output_ids2: Array | None = None) -> Array:
		if output_ids is None:
			if output_ids2 is not None:
				raise ValueError("`output_ids2` requires `output_ids`.")
			x2 = x1 if x2 is None else x2
			output_ids = jnp.repeat(jnp.arange(self.n_outputs), x1.shape[0])
			output_ids2 = jnp.repeat(jnp.arange(self.n_outputs), x2.shape[0])
			x1 = jnp.tile(x1, (self.n_outputs, 1))
			x2 = jnp.tile(x2, (self.n_outputs, 1))
			return self.engine.__call__(self, x1, x2, output_ids, output_ids2)

		output_ids = jnp.asarray(output_ids)
		if output_ids.shape[0] != x1.shape[0]:
			raise ValueError(
				f"`output_ids` must have length {x1.shape[0]} (= len(x1)), got {output_ids.shape[0]}.")

		symmetric = x2 is None
		if symmetric:
			x2, output_ids2 = x1, output_ids
		elif output_ids2 is None:
			raise ValueError("`output_ids2` is required when `x2` is given.")
		else:
			output_ids2 = jnp.asarray(output_ids2)
			if output_ids2.shape[0] != x2.shape[0]:
				raise ValueError(
					f"`output_ids2` must have length {x2.shape[0]} (= len(x2)), got {output_ids2.shape[0]}.")

		return self.engine.__call__(self, x1, x2, output_ids, output_ids2)

	@classmethod
	def from_paper_parameters(cls, S: float | Array, P_inv: Array, lambda_inv: float | Array,
	                           **kwargs) -> ConvolutionKernel:
		"""Build a kernel from the parametrisation of Alvarez & Lawrence (2011), natural (not log) scale.

		Always produces an ``ard=True`` kernel, since the paper's parameters are inherently
		per-output-per-dimension.

		:param S: amplitude of each output, shape ``(n_outputs,)``
		:param P_inv: smoothing-window variance of each output/dimension, shape
		    ``(n_outputs, n_features)``
		:param lambda_inv: latent field variance of each dimension, shape ``(n_features,)``
		"""
		S, P_inv = jnp.atleast_1d(jnp.asarray(S)), jnp.asarray(P_inv)
		a = P_inv + 0.5 * jnp.asarray(lambda_inv)
		variance = S ** 2 * jnp.prod(2 * jnp.pi * 2 * a, axis=-1) ** -0.5
		return cls(variance=variance, bandwidth=a, ard=True, **kwargs)

	def replace(self, variance: None | float | Array = None,
	            bandwidth: None | float | Array = None, **kwargs) -> ConvolutionKernel:
		out = self
		if variance is not None:
			variance = jnp.asarray(variance)
			if jnp.any(variance <= 0):
				raise ValueError("`variance` must be positive.")
			out = eqx.tree_at(
				lambda k: k._variance, out,
				jnp.broadcast_to(out._variance_parametrisation.wrap(variance), out._variance.shape))
		if bandwidth is not None:
			bandwidth = jnp.asarray(bandwidth)
			if jnp.any(bandwidth <= 0):
				raise ValueError("`bandwidth` must be positive.")
			out = eqx.tree_at(
				lambda k: k._bandwidth, out,
				jnp.broadcast_to(out._bandwidth_parametrisation.wrap(bandwidth), out._bandwidth.shape))
		return out

	def __str__(self):
		return f"Convolution(n_outputs={self.n_outputs}, ard={self.ard})"
