import jax.numpy as jnp


def assemble(groups, results_path):

    H = jnp.zeros(jnp.load(f"{results_path}/H0.npy").shape)
    W = jnp.zeros(jnp.load(f"{results_path}/W0.npy").shape)
    B = jnp.zeros(H.shape)

    for group_index, group in enumerate(groups):
        indices = [component_description.index for component_description in group]
        H = H.at[indices, :].set(
                        jnp.load(f"{results_path}/H{group_index}.npy")[indices, :])
        B = B.at[indices, :].set(
                        jnp.load(f"{results_path}/B{group_index}.npy")[indices, :])
        W = W.at[:, indices].set(
                        jnp.load(f"{results_path}/W{group_index}.npy")[:, indices])
    return H, W, B
