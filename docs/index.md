# Kernax Documentation

Welcome to Kernax documentation!

## Overview

Kernax is a JAX-based kernel library for Gaussian Processes with automatic differentiation, JIT compilation, and composable kernel operations.

## Quick Links

- [Installation](installation.md)
- [Getting Started](getting_started.md)
- [API Reference](api/index.md)
- [Examples](examples/index.md)

## Features

- **Fast JIT-compiled computations** using JAX
- **Automatic dimension handling** for various input shapes
- **NaN-aware computations** for padded/masked data
- **Composable kernels** through operator overloading
- **Distinct hyperparameters per batch** for multi-task learning
- **PyTree integration** for JAX transformations

## Contents

```{toctree}
:maxdepth: 2

installation
getting_started
api/index
examples/index
contributing
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`