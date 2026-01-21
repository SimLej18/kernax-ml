"""
kernax-ml: A JAX-based kernel library for Gaussian Processes.

kernax-ml provides a collection of kernel functions (covariance functions) for
Gaussian Process models, with support for automatic differentiation, JIT
compilation, and composable kernel operations.
"""

__version__ = "0.1.3-alpha"
__author__ = "S. Lejoly"
__email__ = "simon.lejoly@unamur.be"
__license__ = "MIT"

from .AbstractKernel import AbstractKernel, StaticAbstractKernel
from .ConstantKernel import ConstantKernel, StaticConstantKernel
from .LinearKernel import LinearKernel, StaticLinearKernel
from .MaternKernels import (
	Matern12Kernel,
	Matern32Kernel,
	Matern52Kernel,
	StaticMatern12Kernel,
	StaticMatern32Kernel,
	StaticMatern52Kernel,
)
from .OperatorKernels import OperatorKernel, ProductKernel, SumKernel
from .PeriodicKernel import PeriodicKernel, StaticPeriodicKernel
from .RationalQuadraticKernel import RationalQuadraticKernel, StaticRationalQuadraticKernel
from .SEKernel import SEKernel, StaticSEKernel
from .WrapperKernels import (
	ActiveDimsKernel,
	ARDKernel,
	BatchKernel,
	DiagKernel,
	ExpKernel,
	LogKernel,
	NegKernel,
	WrapperKernel,
	BlockKernel,
)

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
	"StaticSEKernel",
	"SEKernel",
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
	"BatchKernel",
	"ActiveDimsKernel",
	"ARDKernel",
	"BlockKernel",
]
