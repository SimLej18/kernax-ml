"""
Kernax: A JAX-based kernel library for Gaussian Processes.

Kernax provides a collection of kernel functions (covariance functions) for
Gaussian Process models, with support for automatic differentiation, JIT
compilation, and composable kernel operations.
"""

__version__ = "0.1.0"
__author__ = "S. Lejoly"
__email__ = "simon.lejoly@unamur.be"
__license__ = "MIT"

from .AbstractKernel import StaticAbstractKernel, AbstractKernel
from .RBFKernel import StaticRBFKernel, RBFKernel
from .LinearKernel import StaticLinearKernel, LinearKernel
from .MaternKernels import StaticMatern12Kernel, Matern12Kernel
from .MaternKernels import StaticMatern32Kernel, Matern32Kernel
from .MaternKernels import StaticMatern52Kernel, Matern52Kernel
from .SEMagmaKernel import StaticSEMagmaKernel, SEMagmaKernel
from .PeriodicKernel import StaticPeriodicKernel, PeriodicKernel
from .RationalQuadraticKernel import StaticRationalQuadraticKernel, RationalQuadraticKernel
from .ConstantKernel import StaticConstantKernel, ConstantKernel
from .OperatorKernels import OperatorKernel, SumKernel, ProductKernel
from .WrapperKernels import WrapperKernel, NegKernel, ExpKernel, LogKernel, DiagKernel

__all__ = [
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Base classes
    "StaticAbstractKernel",
    "AbstractKernel",
    # Base kernels
    "StaticRBFKernel",
    "RBFKernel",
    "StaticSEMagmaKernel",
    "SEMagmaKernel",
    "StaticConstantKernel",
    "ConstantKernel",
    "StaticLinearKernel",
    "LinearKernel",
    "StaticPeriodicKernel",
    "PeriodicKernel",
    "StaticRationalQuadraticKernel",
    "RationalQuadraticKernel",
    # Matern family
    "StaticMatern12Kernel",
    "Matern12Kernel",
    "StaticMatern32Kernel",
    "Matern32Kernel",
    "StaticMatern52Kernel",
    "Matern52Kernel",
    # Composite kernels
    "OperatorKernel",
    "SumKernel",
    "ProductKernel",
    # Wrapper kernels
    "WrapperKernel",
    "NegKernel",
    "ExpKernel",
    "LogKernel",
    "DiagKernel",
]
