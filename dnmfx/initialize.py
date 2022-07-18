import jax


def initialize_normal(num_components, num_frames, component_size, random_seed):

    key = jax.random.PRNGKey(random_seed)

    # create H, B, and W, such that X[t,s] ≈ W[t,n]@H[n,s] + 1[t,n]@B[n,s]
    #                                      = X_hat[t,s]

    key, subkey = jax.random.split(key)
    H_logits = jax.random.normal(
        subkey,
        shape=(num_components, component_size))

    key, subkey = jax.random.split(key)
    B_logits = jax.random.normal(
        subkey,
        shape=(num_components, component_size))

    _, subkey = jax.random.split(key)
    W_logits = jax.random.normal(
        subkey,
        shape=(num_frames, num_components))

    return H_logits, W_logits, B_logits
