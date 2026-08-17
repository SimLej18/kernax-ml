"""
Structural type aliases for "anything that behaves like a kernel/mean", including
wrapped and composed ones (``ARDKernel``, ``ICMKernel``, ``SumModule``, ...).

``AbstractKernel``/``AbstractMean`` alone are too narrow for type hints: a wrapper
(``AbstractWrapperModule``, e.g. ``ARDKernel``, ``BatchModule``) or an operator
(``AbstractOperatorModule``, e.g. ``SumModule``) is never an ``AbstractKernel`` or
``AbstractMean`` itself, since it composes one instead of implementing ``pairwise``/
``scalar_mean`` directly. ``KernelLike``/``MeanLike`` cover the full structural family so a
function can accept ``ICMKernel`` without also silently accepting ``BlockMean`` -- wrapper
and operator generics (see ``kernax.module.ModuleT``) keep the two families apart.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
	from .AbstractKernel import AbstractKernel  # noqa: F401
	from .AbstractMean import AbstractMean  # noqa: F401
	from .multioutput.LMCKernel import LMCKernel  # noqa: F401
	from .operators.AbstractOperatorModule import AbstractOperatorModule  # noqa: F401
	from .wrappers.WrapperModule import AbstractWrapperModule  # noqa: F401

KernelLike: TypeAlias = Union[
	"AbstractKernel",
	"LMCKernel",
	"AbstractWrapperModule[KernelLike]",
	"AbstractOperatorModule[KernelLike]",
]
"""A kernel, or any wrapper/operator/LMC composition of kernels."""

MeanLike: TypeAlias = Union[
	"AbstractMean",
	"AbstractWrapperModule[MeanLike]",
	"AbstractOperatorModule[MeanLike]",
]
"""A mean function, or any wrapper/operator composition of mean functions."""
