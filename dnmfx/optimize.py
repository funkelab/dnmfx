from .groups import get_groups
from .initialize import initialize_normal
from .log import Log
from .loss import l2_loss_grad
from .utils import sigmoid
from datetime import datetime
from tqdm import tqdm
import jax
import jax.numpy as jnp
import random
from timeit import default_timer as timer


def dnmf(
        sequence,
        component_descriptions,
        parameters,
        H_logits,
        W_logits,
        B_logits):
    """Perform distributed NMF on the given sequence.

    Args:

        sequence (array-like, shape `(t, [z,] y, x)`):

            The raw data (usually referred to as 'X') to factorize into `X =
            H@W`, where `H` is an array of the estimated components and `W` is
            their activity over time.

        component_descriptions (list of :class:`ComponentDescription`):

            The bounding boxes and indices of the components to estimate.

        parameters (:class:`Parameters`):

            Parameters to control the optimization.

        H_logits (array-like, shape `(k, w*h)`):

            Array of the estimated components.

        W_logits (array-like, shape `(t, k)`):

            Array of the activities of the estimated components.

        B_logits (array-like, shape `(k, w*h)`):

            Array of the background of the estimate components.
   """
    log = Log()
    l2_loss_grad_jit = jax.jit(l2_loss_grad,
                               static_argnames=['component_description'])
    update_jit = jax.jit(update)
    aggregate_loss = 0

    for iteration in tqdm(range(parameters.max_iteration)):

        # pick a random component
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
            l2_loss_grad_jit(
                H_logits,
                W_logits,
                B_logits,
                x,
                component_description,
                frame_indices)

        aggregate_loss += loss

        if iteration % parameters.log_every == 0:

            if iteration == 0: average_loss = loss
            else: average_loss = float(aggregate_loss/parameters.log_every)

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

            elif iteration > 0:
                log.log_iteration(iteration, average_loss)

            if average_loss < parameters.min_loss:
                print(f"Optimization converged ({average_loss}<{parameters.min_loss})")
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

    slices = bounding_box.to_slices()
    x = jnp.array([sequence[(t,) + slices] for t in frames])
    x = x.reshape(-1, *bounding_box.shape)

    return x


def update(H, W, B, grad_H, grad_W, grad_B, step_size):

    H = H - step_size * grad_H
    W = W - step_size * grad_W
    B = B - step_size * grad_B

    return H, W, B
