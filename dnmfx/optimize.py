from .log import Log
from .loss import nmf_loss_grad
from .utils import sigmoid
from tqdm import tqdm
import jax
import jax.numpy as jnp
import random


def dnmf(
        dataset,
        component_descriptions,
        parameters,
        H_logits,
        W_logits,
        B_logits):
    """Perform distributed NMF on the given sequence.

    Args:

       dataset (:class:`Dataset`):
            Dataset to be fitted; should have a `sequence` array of shape `(t,
            [[z,], y,] x)`.

       component_descriptions (list of :class:`ComponentDescription`):
            The bounding boxes and indices of the components to estimate.

        parameters (:class:`Parameters`):
            Parameters to control the optimization.

        H_logits (array-like, shape `(k, w*h)`):
            Array of the estimated components to be optimized.

        W_logits (array-like, shape `(t, k)`):
            Array of the activities of the estimated components to be
            optimized.

        B_logits (array-like, shape `(k, w*h)`):
            Array of the background of the estimate components to be optimized.

    Returns:

        The optimization results (i.e. H, W, B).
    """

    log = Log()
    nmf_loss_grad_jit = jax.jit(
        nmf_loss_grad,
        static_argnums=(4,))
    update_jit = jax.jit(update)
    aggregate_loss = 0

    sequence = dataset.sequence

    for iteration in tqdm(range(parameters.max_iteration)):

        # pick a random component
        random.seed(parameters.random_seed + iteration)
        component_description = random.sample(component_descriptions, 1)[0]
        component_bounding_box = component_description.bounding_box

        num_frames = sequence.shape[0]
        # pick a random subset of frames
        frame_indices = tuple(random.sample(
            list(range(num_frames)),
            parameters.batch_size))

        # gather the sequence data for those components/frames
        x = get_x(sequence, frame_indices, component_bounding_box)

        # compute the current loss and gradient
        loss, (grad_H_logits, grad_W_logits, grad_B_logits) = \
            nmf_loss_grad_jit(
                H_logits,
                W_logits,
                B_logits,
                x,
                component_description,
                frame_indices,
                parameters.l1_weight)

        aggregate_loss += loss

        if iteration % parameters.log_every == 0:

            if iteration == 0:
                average_loss = loss
            else:
                average_loss = float(aggregate_loss/parameters.log_every)

            aggregate_loss = 0

            # log gradients after the 1st iteration
            if iteration == 0 and parameters.log_gradients:
                log.log_iteration(
                            iteration,
                            average_loss,
                            grad_H_logits,
                            grad_W_logits,
                            grad_B_logits,
                            H_logits,
                            W_logits,
                            B_logits)

            elif not parameters.log_gradients:
                log.log_iteration(iteration,
                                  average_loss)

            if average_loss < parameters.min_loss:
                print(f"Optimization converged: \
                        {average_loss}<{parameters.min_loss}")
                break

        # update current estimate
        H_logits, W_logits, B_logits = update_jit(
            H_logits,
            W_logits,
            B_logits,
            grad_H_logits,
            grad_W_logits,
            grad_B_logits,
            parameters.step_size)

    return sigmoid(H_logits), sigmoid(W_logits), sigmoid(B_logits), log


def get_x(sequence, frames, bounding_box):
    """Extract the region defined by the bounding box from the given sequence.

    Args:
        sequence (array-like, shape `(t, [z,] y, x)`):
            The raw data (usually referred to as 'X') to factorize into `X =
            H@W`, where `H` is an array of the estimated components and `W` is
            their activity over time.

        frames (list):
            A list of frame indices of length the batch size.

        bounding_box (:class: `funlib.geometry.Roi`):
            Bounding box of :class: `funlib.geometry.Roi` that defines a
            rectangular region.

    Returns:
         A region defined by the bounding box from the given sequence.
    """

    slices = bounding_box.to_slices()
    x = jnp.array([sequence[(t,) + slices] for t in frames])
    x = x.reshape(-1, *bounding_box.shape)

    return x


def update(H_logits, W_logits, B_logits, grad_H, grad_W, grad_B, step_size):
    """Update matrix factors H, W, B by their gradients and update step size.

    Args:
        H_logits (array-like, shape `(k, w*h)`):
            Array of the estimated components to be updated.

        W_logits (array-like, shape `(t, k)`):
            Array of the activities of the estimated components to be updated.

        B_logits (array-like, shape `(k, w*h)`):
            Array of the background of the estimate components to be updated.

        grad_H (array-like, shape `(k, w*h)`):
            Array of gradients of the optimization objective wrt. `H_logits`.

        grad_W (array-like, shape `(t, k)`):
            Array of gradients of the optimization objective wrt. `W_logits`.

        grad_B (array-like, shape `(k, w*h)`):
            Array of gradients of the optimization objective wrt. `B_logits`.

        step_size (float):
            Size of updating step.

    Returns:
        Updated `H_logits`, `W_logits`, `B_logits`.
    """

    H_logits = H_logits - step_size * grad_H
    W_logits = W_logits - step_size * grad_W
    B_logits = B_logits - step_size * grad_B

    return H_logits, W_logits, B_logits
