# The Sharp Bits

At its core, Kernax is just a bunch of kernel formulas wrapped inside JAX transformations, inside Equinox Modules, aka pytrees.

Although this leads to nice and efficient abstractions, it also means that you, as the end user, might be aware of a 
few common pitfalls to get comfortable using Kernax.

## External references

Before diving into Kernax-specific pitfalls, it is worth familiarising yourself with:

- [JAX sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) — jit, grad, vmap, random keys, in-place mutations
- [All of Equinox](https://docs.kidger.site/equinox/all-of-equinox/) — how Modules work as pytrees, filtering, JIT on Modules
- [GPJax sharp bits](https://docs.jaxgaussianprocesses.com/sharp_bits/) — especially PRNG handling, bijectors, and PSD-ness of Gram matrices
