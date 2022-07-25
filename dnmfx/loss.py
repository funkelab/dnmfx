from .utils import sigmoid
import jax
import jax.numpy as jnp


def l2_loss(
        H_logits,
        W_logits,
        B_logits,
        x,
        component_description,
        frame_indices):

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

    return jnp.linalg.norm(x - x_hat)


l2_loss_grad = jax.value_and_grad(l2_loss, argnums=(0, 1, 2))

def get_x_hat(H_logits, W_logits, B_logits, component_description, frames):

    i = component_description.index
    bb_i = component_description.bounding_box

    w = sigmoid(W_logits[frames, i])
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

        w = sigmoid(W_logits[frames, j])
        h = sigmoid(H_logits[slices_j])
        b = sigmoid(B_logits[slices_j])

        x_hat = x_hat.at[slices_i].add(jnp.outer(w, h).reshape(-1, *h.shape) + b)

    return x_hat
