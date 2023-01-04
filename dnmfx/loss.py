from .utils import sigmoid
import jax
import jax.numpy as jnp


def nmf_loss(
        H_logits,
        W_logits,
        B_logits,
        x,
        component_description,
        frame_indices,
        l1_weight):

    reconstruction_loss = l2_loss(
        H_logits,
        W_logits,
        B_logits,
        x,
        component_description,
        frame_indices)

    regularizer_loss = l1_loss(
        H_logits[component_description.index],
        W_logits[component_description.index])

    return reconstruction_loss + l1_weight * regularizer_loss


nmf_loss_grad = jax.value_and_grad(nmf_loss, argnums=(0, 1, 2))


def l2_loss(
        H_logits,
        W_logits,
        B_logits,
        x,
        component_description,
        frame_indices):
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

        component_description (:class: `ComponentDescription`):
            The bounding box, index, and overlapping components of the given
            component `x`.

        frame_indices (list):
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
            component_description,
            frame_indices)

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


def get_x_hat(H_logits, W_logits, B_logits, component_description, frames):
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

        component_description (:class: `ComponentDescription`):
            The bounding box, index, and overlapping components of some
            component `x`.

        frames (list):
            A list of frame indices of length the batch size.

    Returns:

        Reconstructed x̂_c.
    """

    i = component_description.index
    bb_i = component_description.bounding_box
    w = sigmoid(W_logits[i, frames])
    h = sigmoid(H_logits[i])
    b = sigmoid(B_logits[i])

    x_hat = jnp.outer(w, h).reshape(-1, *h.shape) + b
    x_hat = x_hat.reshape(-1, *bb_i.shape)

    for overlap in component_description.overlapping_components:

        j = overlap.index
        bb_j = overlap.bounding_box

        intersection = bb_i.intersect(bb_j)
        intersection_in_c_i = intersection - bb_i.get_begin()
        intersection_in_c_j = intersection - bb_j.get_begin()

        slices_i = (slice(None),) + intersection_in_c_i.to_slices()
        slices_j = (j,) + intersection_in_c_j.to_slices()

        H_logits = H_logits.reshape(-1, *bb_i.shape)
        B_logits = B_logits.reshape(-1, *bb_i.shape)

        w = sigmoid(W_logits[j, frames])
        h = sigmoid(H_logits[slices_j])
        b = sigmoid(B_logits[slices_j])

        x_hat = \
            x_hat.at[slices_i].add(jnp.outer(w, h).reshape(-1, *h.shape) + b)

    return x_hat
