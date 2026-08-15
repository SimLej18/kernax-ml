# %% [markdown]
# # Multi-output GP -- Linear Model of Coregionalisation (LMC)
#
# A full multi-output GP built from three `kernax.multioutput` pieces:
#
# - `LMCKernel` correlates the `P` outputs through `Q` independent latent kernels, each
#   paired with its own free `(P, P)` coregionalisation matrix `B_q = W_q W_qᵀ`, summed:
#   `K(x1, x2) = Σ_q B_q ⊗ k_q(x1, x2)`. Unlike ICM, each component can have a different
#   kernel class/hyperparameters -- e.g. a short and a long length-scale, mixed per output.
# - `BlockDiagKernel` adds independent per-output observation noise on top (`+`, like any
#   other kernax kernel).
# - `BlockMean` gives every output its own mean function.
#
# The config cell below picks: isotopic (all outputs on one shared grid) vs heterotopic
# (each output on its own points) data, shared vs per-output hyperparameters for the
# noise/mean terms, and which latent kernels/mean are used underneath.

# %%
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from kernax import SEKernel, WhiteNoiseKernel
from kernax.mask import create_mask
from kernax.means import ConstantMean
from kernax.multioutput import BlockDiagKernel, BlockMean, LMCKernel

# %% [markdown]
# ## Config

# %%
N_OUTPUTS = 3
ISOTOPIC = False             # True: every output observed on the same grid. False: heterotopic, own points per output.
SHARED_OUTPUT_HPS = False   # True: the noise/mean terms share one value across outputs. False: one value per output.
COMPONENT_KERNELS = [SEKernel(length_scale=1.5), SEKernel(length_scale=0.2)]  # k_q, one per latent component
COMPONENT_RANKS = [2, 1]    # rank of each B_q = W_q W_qᵀ, one per component -- len must match COMPONENT_KERNELS
# mean template; swapping the class means updating the "constant" field name in the mask below too
INNER_MEAN = ConstantMean(0.0)
NOISE = 0.01
SEED = 0

# %% [markdown]
# ## Build the GP: `LMCKernel + BlockDiagKernel` for the kernel, `BlockMean` for the mean

# %%
key = jr.PRNGKey(SEED)
key, *w_keys = jr.split(key, len(COMPONENT_KERNELS) + 1)

# Non-trivial W_q per component, same reasoning as the ICM notebook: eye(P, rank) would
# leave some outputs uncorrelated whenever rank < P, which defeats the coregionalisation demo.
Ws = [jr.normal(k, (N_OUTPUTS, r)) * 0.7 for k, r in zip(w_keys, COMPONENT_RANKS, strict=True)]
lmc = LMCKernel(kernels=COMPONENT_KERNELS, coregionalization_matrices=Ws)

noise_axes = None if SHARED_OUTPUT_HPS else create_mask(WhiteNoiseKernel(1.0), noise=0)
noise = BlockDiagKernel(WhiteNoiseKernel(NOISE), n_outputs=N_OUTPUTS, output_hps_in_axes=noise_axes)
if not SHARED_OUTPUT_HPS:
	noise = noise.replace(noise=NOISE * jnp.array([0.5, 1.0, 2.0]))

kernel = lmc + noise

mean_axes = None if SHARED_OUTPUT_HPS else create_mask(ConstantMean(0.0), constant=0)
mean = BlockMean(INNER_MEAN, n_outputs=N_OUTPUTS, output_hps_in_axes=mean_axes)
if not SHARED_OUTPUT_HPS:
	mean = mean.replace(constant=jnp.array([-1.5, 0.0, 1.5]))

print("kernel:", kernel)
print("mean:  ", mean)

# %% [markdown]
# ## Data: isotopic or heterotopic, selected by the config above
#
# Both branches produce the same two objects downstream code relies on: `x_all` (all
# points, any output order) and `output_ids` (which output each row of `x_all` belongs to).
# That is the whole point of the `output_ids` convention -- the GP built above does not
# change at all between the two branches.

# %%
key, *output_keys = jr.split(key, N_OUTPUTS + 1)

if ISOTOPIC:
	x_shared = jnp.linspace(0.0, 10.0, 60)[:, None]
	xs_per_output = [x_shared] * N_OUTPUTS
else:
	sizes = (100, 200, 300)
	xs_per_output = [
		jnp.sort(jr.uniform(k, (n, 1), minval=0.0, maxval=10.0))
		for k, n in zip(output_keys, sizes, strict=True)
	]

x_all = jnp.concatenate(xs_per_output)
output_ids = jnp.concatenate([jnp.full(xo.shape[0], o) for o, xo in enumerate(xs_per_output)])

# %% [markdown]
# ## Prior sample
#
# One joint draw over every output at once, using the same `(mean, kernel)` pair and the
# same `output_ids` regardless of ISOTOPIC/SHARED_OUTPUT_HPS -- this is the "ground truth"
# the rest of the notebook fits against.

# %%
K_prior = kernel(x_all, output_ids=output_ids, output_ids2=output_ids)
m_prior = mean(x_all, output_ids=output_ids)
L_prior = jnp.linalg.cholesky(K_prior + 1e-6 * jnp.eye(K_prior.shape[0]))

key, sample_key = jr.split(key)
f_true = m_prior + L_prior @ jr.normal(sample_key, (K_prior.shape[0],))

# %% [markdown]
# When the data is isotopic, `output_ids` can be omitted entirely and the kernel/mean take
# the shared-grid shortcut instead (dense Kronecker product, no gather): same result,
# cheaper to compute.

# %%
if ISOTOPIC:
	K_shortcut = kernel(x_shared)
	m_shortcut = mean(x_shared)
	assert jnp.allclose(K_shortcut, K_prior, atol=1e-5)
	assert jnp.allclose(m_shortcut, m_prior, atol=1e-5)
	print("shared-grid shortcut (no output_ids) matches the output_ids route:", True)

# %% [markdown]
# ## Observations and posterior
#
# A noisy subset of the prior sample (every 4th point per output) plays the role of
# observed data. The posterior is the textbook GP regression formula, computed once over
# every output jointly -- `kernel`/`mean` don't need to know which points are train vs
# test, only their `output_ids`.

# %%
train_mask = jnp.zeros(x_all.shape[0], dtype=bool)
for o in range(N_OUTPUTS):
	idx = jnp.where(output_ids == o)[0]
	train_mask = train_mask.at[idx[::4]].set(True)

key, noise_key = jr.split(key)
obs_noise_std = 0.1
y_train = f_true[train_mask] + obs_noise_std * jr.normal(noise_key, (int(train_mask.sum()),))
x_train, ids_train = x_all[train_mask], output_ids[train_mask]
x_test, ids_test = x_all, output_ids  # posterior evaluated back on the full reference grid

K_train = kernel(x_train, output_ids=ids_train, output_ids2=ids_train)
K_train += obs_noise_std ** 2 * jnp.eye(K_train.shape[0])
K_cross = kernel(x_test, x_train, output_ids=ids_test, output_ids2=ids_train)
K_test = kernel(x_test, output_ids=ids_test, output_ids2=ids_test)

L = jnp.linalg.cholesky(K_train)
alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y_train - mean(x_train, output_ids=ids_train)))
v = jnp.linalg.solve(L, K_cross.T)

posterior_mean = mean(x_test, output_ids=ids_test) + K_cross @ alpha
posterior_cov = K_test - v.T @ v
posterior_std = jnp.sqrt(jnp.clip(jnp.diag(posterior_cov), 0.0))

# %% [markdown]
# ## Plot: one panel per output

# %%
fig, axes = plt.subplots(1, N_OUTPUTS, figsize=(5 * N_OUTPUTS, 4), sharey=True)
for o, ax in enumerate(axes):
	sel, sel_train = output_ids == o, ids_train == o
	order = jnp.argsort(x_all[sel, 0])

	xo = x_all[sel, 0][order]
	ax.plot(xo, f_true[sel][order], "k--", label="prior sample (truth)")
	ax.plot(xo, posterior_mean[sel][order], color="C0", label="posterior mean")
	ax.fill_between(
		xo,
		(posterior_mean[sel] - 2 * posterior_std[sel])[order],
		(posterior_mean[sel] + 2 * posterior_std[sel])[order],
		color="C0", alpha=0.2, label="±2 std",
	)
	ax.scatter(x_train[sel_train, 0], y_train[sel_train], color="k", zorder=5, label="observations")
	ax.set_title(f"output {o}")
	ax.set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle(
	f"LMC multi-output GP -- ISOTOPIC={ISOTOPIC}, SHARED_OUTPUT_HPS={SHARED_OUTPUT_HPS}"
)
fig.tight_layout()
fig.show()
