from .utils import sigmoid
import jax
import jax.numpy as jnp


def nmf_loss(
        H_logits,
        W_logits,
        B_logits,
        xs,
        H_index_maps,
        W_index_maps,
        frames,
        l1_weight):

    vmap_loss = jax.vmap(
        l2_loss,
        in_axes=(None, None, None, 0, 0, 0, None)
    )

    reconstruction_losses = vmap_loss(
        H_logits,
        W_logits,
        B_logits,
        xs,
        H_index_maps,
        W_index_maps,
        frames)

    reconstruction_loss = jnp.mean(reconstruction_losses)

    component_indices = W_index_maps[:, 0]
    regularizer_loss = l1_loss(
        H_logits[component_indices],
        W_logits[component_indices])

    return reconstruction_loss + l1_weight * regularizer_loss


nmf_loss_grad = jax.value_and_grad(nmf_loss, argnums=(0, 1, 2))


def l2_loss(
        H_logits,
        W_logits,
        B_logits,
        x,
        H_index_map,
        W_index_map,
        frames):
    """Compute the L2 distance between data from a single component and
    reconstruction from optimization results.

    Args:

        H_logits (array-like, shape `(k, w*h)`):
             Array of the estimated components.

        W_logits (array-like, shape `(k, t)`):
            Array of the activities of the estimated components.

        B_logits (array-like, shape `(k, w*h)`):
            Array of the background of the estimate components.

        x (array-like, shape `(w, h)`):
            Data from a single component.

        H_index_map (array-like, shape `(n, w*h)`):
            The indices into the H array that make up this component, including
            `n-1` overlapping components.

        W_index_map (array-like, shape `(n,)`):
            The indices into the W array that correspond to the components in
            `H_index_map`.

        frames (list):
            A list of frame indices of length the batch size.

    Returns:

        L2 distance between x and reconstruction `H_logits`, `W_logits`,
        `B_logits`.
    """

    assert len(H_logits.shape) == 2
    assert len(W_logits.shape) == 2
    assert len(B_logits.shape) == 2

    # get the current estimate for what x would look like (i.e., x_hat)
    x_hat = get_x_hat(
            H_logits,
            W_logits,
            B_logits,
            H_index_map,
            W_index_map,
            frames)

    l2_loss = jnp.linalg.norm(x - x_hat)

    return l2_loss


def l1_loss(H_logits, W_logits):

    l1_loss_H = jnp.linalg.norm(
        sigmoid(H_logits),
        ord=1)

    l1_loss_W = jnp.linalg.norm(
        sigmoid(W_logits),
        ord=1)

    return l1_loss_H + l1_loss_W


def get_x_hat(H_logits, W_logits, B_logits, H_index_map, W_index_map, frames):
    """Estimate reconstruction of a single component from array of estimated
    components, traces, and backgrounds; suppose the component to be estimated
    is c and denote every of its overlapping component as c', we reconstruct
    x_c as the following:

                x̂_c = B_c + W_c * H_c + Σ [B_c' + W_c' + H_c'].

    Args:

        H_logits (array-like, shape `(k, w*h)`):
            Array of the estimated components.

        W_logits (array-like, shape `(t, k)`):
            Array of the activities of the estimated components.

        B_logits (array-like, shape `(k, w*h)`):
            Array of the background of the estimate components.

        H_index_map (array-like, shape `(n, w*h)`):
            The indices into the H array that make up this component, including
            `n-1` overlapping components.

        W_index_map (array-like, shape `(n,)`):
            The indices into the W array that correspond to the components in
            `H_index_map`.

        frames (list):
            A list of frame indices of length the batch size.

    Returns:

        Reconstructed x̂_c.
    """

    def get_component(H_indices, W_index):

        h = sigmoid(
            H_logits.at[W_index, H_indices].get(
                mode='fill',
                fill_value=-1e10  # 0 after sigmoid
            )
        )
        b = sigmoid(
            B_logits.at[W_index, H_indices].get(
                mode='fill',
                fill_value=-1e10  # 0 after sigmoid
            )
        )
        w = sigmoid(W_logits[W_index, frames])

        return jnp.outer(w, h) + b.flatten()

    components = jax.vmap(get_component)(H_index_map, W_index_map)
    x_hat = components.sum(axis=0)

    return x_hat
