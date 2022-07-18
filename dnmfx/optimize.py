from .initialize import initialize_normal
from .log import Log
from .loss import l2_loss_grad
from .utils import sigmoid
from datetime import datetime
from tqdm import tqdm
import jax.numpy as jnp
import random


def dnmf(
        sequence,
        component_descriptions,
        parameters,
        log_every=10,
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
    for i in tqdm(range(parameters.max_iteration)):

        # pick a random component
        component_index = random.randint(0, num_components - 1)
        component_description = component_descriptions[component_index]
        component_bounding_box = component_description.bounding_box
        component_indices = (component_index,)

        # pick a random subset of frames
        frame_indices = random.sample(
            list(range(num_frames)),
            parameters.batch_size)

        # gather the sequence data for those components/frames
        x = extract(sequence, frame_indices, component_bounding_box)

        # gather the relevant parts of H, W, and B
        h_logits = H_logits[component_indices, :]
        w_slices = (tuple(frame_indices), component_indices)
        w_logits = W_logits[w_slices]
        w_logits = w_logits.reshape(parameters.batch_size, 1)
        b_logits = B_logits[component_indices, :]

        # compute the current loss and gradient
        loss, (grad_h_logits, grad_w_logits, grad_b_logits) = \
            l2_loss_grad(h_logits, w_logits, b_logits, x)

        # log the loss
        log.add_loss(i, loss)

        if loss < parameters.min_loss:
            print(f"Optimization converged ({loss}<{parameters.min_loss})")
            break

        # update current estimate
        h_logits, w_logits, b_logits = update(
            h_logits,
            w_logits,
            b_logits,
            grad_h_logits,
            grad_w_logits,
            grad_b_logits,
            parameters.step_size)

        # replace global estimates with updates
        H_logits = H_logits.at[component_indices, :].set(h_logits)
        W_logits = W_logits.at[w_slices].set(w_logits.flatten())
        B_logits = B_logits.at[component_indices, :].set(b_logits)

    return sigmoid(H_logits), sigmoid(W_logits), sigmoid(B_logits), log


def extract(sequence, frames, bounding_box):

    slices = bounding_box.to_slices()
    x = jnp.array([sequence[(t,) + slices] for t in frames])
    x = x.reshape(len(frames), -1)

    return x


def update(H, W, B, grad_H, grad_W, grad_B, step_size):

    H = H - step_size * grad_H
    W = W - step_size * grad_W
    B = B - step_size * grad_B

    return H, W, B
