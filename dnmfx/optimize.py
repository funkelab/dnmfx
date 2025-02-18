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

    nmf_loss_grad_jit = jax.jit(nmf_loss_grad)
    update_jit = jax.jit(update)

    log = Log()
    aggregate_loss = 0

    sequence = jnp.array(dataset.sequence)
    frames = list(range(sequence.shape[0]))

    progress = tqdm(range(parameters.max_iteration))
    for iteration in progress:

        # pick a random subset of components
        random.seed(parameters.random_seed + iteration)
        batch_components = random.sample(
            component_descriptions,
            min(parameters.batch_components, len(component_descriptions)))

        # pick a random subset of frames
        frame_indices = jnp.array(
            random.sample(frames, min(parameters.batch_frames, len(frames))),
            dtype=jnp.int32)

        # gather the sequence data for those components/frames
        xs = jnp.array([
            get_x(sequence, frame_indices, c.bounding_box)
            for c in batch_components
        ])

        # gather the H and W index maps
        H_index_maps = jnp.array([c.H_index_map for c in batch_components])
        W_index_maps = jnp.array([c.W_index_map for c in batch_components])

        # compute the current loss and gradient
        loss, (grad_H_logits, grad_W_logits, grad_B_logits) = \
            nmf_loss_grad_jit(
                H_logits,
                W_logits,
                B_logits,
                xs,
                H_index_maps,
                W_index_maps,
                frame_indices,
                parameters.l1_weight)

        progress.set_description("batch loss=%.3f" % loss)

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
    x = sequence[frames, *slices]
    return x.reshape(len(frames), -1)


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
