# Installation

## Jax dependency

Kernax requires JAX. Install it first following the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html) — this ensures you pick the right backend (CPU, CUDA, Metal) and a compatible version.

By default, kernax installs the CPU-only Jax implementation.

## Kernax installation

### Pip package

```bash
pip install kernax-ml
```

### From source

```bash
git clone https://github.com/SimLej18/kernax-ml.git
cd kernax-ml
pip install -e .
```
