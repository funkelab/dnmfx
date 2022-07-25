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
    x_hat, x_hat_logits, h_logits, w_logits, b_logits = get_x_hat(
        H_logits,
        W_logits,
        B_logits,
        component_description,
        frame_indices)

    return (jnp.linalg.norm(x - x_hat), (x_hat, x_hat_logits, h_logits, w_logits, b_logits))


l2_loss_grad = jax.value_and_grad(l2_loss, argnums=(0, 1, 2), has_aux=True)

def get_x_hat(H_logits, W_logits, B_logits, component_description, frames):

    i = component_description.index
    bb_i = component_description.bounding_box

    w_logits = W_logits[frames, i]
    h_logits = H_logits[i]
    b_logits = B_logits[i]

    x_hat_logits = jnp.outer(w_logits, h_logits).reshape(-1, *h_logits.shape) + \
                            b_logits

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

        w_logits = W_logits[frames, j]
        h_logits = H_logits[slices_j]
        b_logits = B_logits[slices_j]

        x_hat_logits = x_hat_logits.reshape(-1, *bb_i.shape)
        x_hat_logits = x_hat_logits.at[slices_i].add(
                    jnp.outer(w_logits, h_logits).reshape(-1, *h_logits.shape) + \
                    b_logits)

    x_hat = sigmoid(x_hat_logits).reshape(-1, *bb_i.shape)

    return x_hat, x_hat_logits, h_logits, w_logits, b_logits

