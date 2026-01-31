"""
kernax-ml: A JAX-based kernel library for Gaussian Processes.

kernax-ml provides a collection of kernel functions (covariance functions) for
Gaussian Process models, with support for automatic differentiation, JIT
compilation, and composable kernel operations.
"""

__version__ = "0.4.0-alpha"
__author__ = "S. Lejoly"
__email__ = "simon.lejoly@unamur.be"
__license__ = "MIT"

from .AbstractKernel import AbstractKernel, StaticAbstractKernel

# Import configuration system
from .config import config

# Import transformation utilities
from . import transforms

# Import operator kernels
from .operators import (
	OperatorKernel,
	ProductKernel,
	SumKernel,
)

# Import stationary kernels
from .stationary import (
	Matern12Kernel,
	Matern32Kernel,
	Matern52Kernel,
	PeriodicKernel,
	RationalQuadraticKernel,
	RBFKernel,
	SEKernel,
	StaticMatern12Kernel,
	StaticMatern32Kernel,
	StaticMatern52Kernel,
	StaticPeriodicKernel,
	StaticRationalQuadraticKernel,
	StaticSEKernel,
)

# Import dot-product kernels
from .dotproduct import (
	LinearKernel,
	PolynomialKernel,
	SigmoidKernel,
	StaticLinearKernel,
	StaticPolynomialKernel,
	StaticSigmoidKernel,
)

# Import other kernels
from .other import (
	ConstantKernel,
	StaticConstantKernel,
	WhiteNoiseKernel,
)

# Import wrapper kernels
from .wrappers import (
	ActiveDimsKernel,
	ARDKernel,
	BatchKernel,
	BlockDiagKernel,
	BlockKernel,
	DiagKernel,
	ExpKernel,
	LogKernel,
	NegKernel,
	WrapperKernel,
)

__all__ = [
	# Package metadata
	"__version__",
	"__author__",
	"__email__",
	"__license__",
	# Configuration
	"config",
	# Transformations
	"transforms",
	# Base classes
	"StaticAbstractKernel",
	"AbstractKernel",
	# Base kernels
	"StaticSEKernel",
	"SEKernel",
	"RBFKernel",
	"StaticConstantKernel",
	"ConstantKernel",
	"StaticLinearKernel",
	"LinearKernel",
	"StaticPeriodicKernel",
	"PeriodicKernel",
	"StaticRationalQuadraticKernel",
	"RationalQuadraticKernel",
	"StaticPolynomialKernel",
	"PolynomialKernel",
	"StaticSigmoidKernel",
	"SigmoidKernel",
	"WhiteNoiseKernel",
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
	"BlockDiagKernel",
]
