"""
kernax-ml: A JAX-based kernel library for Gaussian Processes.

kernax-ml provides a collection of kernel functions (covariance functions) for
Gaussian Process models, with support for automatic differentiation, JIT
compilation, and composable kernel operations.
"""

__version__ = "0.5.5-alpha"
__author__ = "S. Lejoly"
__email__ = "simon.lejoly@unamur.be"
__license__ = "MIT"

# Import transformation utilities
from .parametrisations import AbstractParametrisation, LogExpParametrisation
from .module import AbstractModule
from .AbstractKernel import AbstractKernel
from .AbstractMean import AbstractMean

# Import dot-product kernels
from .dotproduct import (
	LinearKernel,
	AffineKernel,
	PolynomialKernel,
	SigmoidKernel,
)

# Import operator modules
from .operators import (
	OperatorModule,
	ProductModule,
	SumModule,
)

# Import other kernels
from .other import (
	ConstantKernel,
	VarianceKernel,
	WhiteNoiseKernel,
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
	FeatureKernel,
)

# Import wrapper modules/kernels
from .wrappers import (
	ActiveDimsModule,
	ARDKernel,
	BatchModule,
	BlockDiagKernel,
	BlockKernel,
	ExpModule,
	LogModule,
	NegModule,
	WrapperModule,
)

# Import HP sampling utilities
from .hp_sampling import sample_hps_from_uniform_priors, sample_hps_from_normal_priors

# Import mask utility
from .mask import create_mask

# Import mean functions
from .means import (
	AffineMean,
	ConstantMean,
	LinearMean,
	ZeroMean,
)

__all__ = [
	# Package metadata
	"__version__",
	"__author__",
	"__email__",
	"__license__",
	# Base classes
	"AbstractModule",
	"AbstractKernel",
	"AbstractMean",
	# Base kernels
	"SEKernel",
	"RBFKernel",
	"ConstantKernel",
	"LinearKernel",
	"AffineKernel",
	"PeriodicKernel",
	"RationalQuadraticKernel",
	"PolynomialKernel",
	"SigmoidKernel",
	"VarianceKernel",
	"WhiteNoiseKernel",
	"FeatureKernel",
	# Matern family
	"Matern12Kernel",
	"Matern32Kernel",
	"Matern52Kernel",
	# Operator modules
	"OperatorModule",
	"SumModule",
	"ProductModule",
	# Wrapper modules/kernels
	"WrapperModule",
	"NegModule",
	"ExpModule",
	"LogModule",
	"ActiveDimsModule",
	"ARDKernel",
	"BatchModule",
	"BlockKernel",
	"BlockDiagKernel",
	# HP sampling
	"sample_hps_from_uniform_priors",
	"sample_hps_from_normal_priors",
	# Parametrisations
	"AbstractParametrisation",
	"LogExpParametrisation",
	# Mask utility
	"create_mask",
	# Mean functions
	"ZeroMean",
	"ConstantMean",
	"LinearMean",
	"AffineMean",
]
