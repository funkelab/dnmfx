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


def dnmf(
        sequence,
        component_descriptions,
        parameters,
        log_every=10,
        log_gradients=False,
        random_seed=None):
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

        log_every (int):

            How often to print iteration statistics.

        random_seed (int):

            A random seed for the initialization of `H` and `W`. If not given,
            a different random seed will be used each time.
    """

    num_frames = sequence.shape[0]
    num_components = len(component_descriptions)
    connected_components = get_groups(component_descriptions)
    print(f"number of connected components: {len(connected_components)}")

    if random_seed is None:
        random_seed = datetime.now().toordinal()

    component_size = None
    for description in component_descriptions:
        size = description.bounding_box.get_size()
        if component_size is not None:
            assert component_size == size, \
                "Only components of the same size are supported for now"
        else:
            component_size = size

    H_logits, W_logits, B_logits = initialize_normal(
        num_components,
        num_frames,
        component_size,
        random_seed)

    log = Log()
    l2_loss_grad_jit = jax.jit(l2_loss_grad,
                               static_argnames=['component_description'])
    update_jit = jax.jit(update)

    for i in tqdm(range(parameters.max_iteration)):

        # pick a random component
        component_index = random.randint(0, num_components - 1)
        component_description = component_descriptions[component_index]
        component_bounding_box = component_description.bounding_box

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

        # log the loss
        if log_gradients:
            log.log_iteration(
                        i,
                        loss,
                        log_gradients,
                        grad_H_logits,
                        grad_W_logits,
                        grad_B_logits,
                        H_logits,
                        W_logits,
                        B_logits)
        else:
            log.log_iteration(i, loss)

        if loss < parameters.min_loss:
            print(f"Optimization converged ({loss}<{parameters.min_loss})")
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
