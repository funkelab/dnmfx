from dnmfx.io import read_dataset
import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(H, W, B, dataset):
    """Get the number of component and component background ID mismatches and
    find the reconstruction error per component and per trace.

    Args:

        H (array-like, shape `(k, w*h)`):
            The factorized matrix that contains all decomposed components.

        W (array-like, shape `(t, k)`):
            The factorized matrix that contains the traces of all decomposed
            components.

        dataset (:class: `Dataset`):
            Dataset to be factorized.

    Returns:

        component_loss (dictionary):
            A dictionary that corresponds each component index (key) to its loss
            (value) that is the L2 distance between the ground truth (if known) and
            component estimation from optmization.

        trace_loss (dictionary):
            A dictionary that corresponds each component index (key) to its loss
            (value) that is the L2 distance between the ground truth (if known) and
            trace estimation from optmization.
    """

    components = dataset.components
    traces = dataset.traces
    sequence = dataset.sequence
    num_components = dataset.num_components
    num_frames = dataset.num_frames
    bounding_boxes = dataset.bounding_boxes
    background = dataset.background
    noises = dataset.noises
    num_component_pixels = components.size
    num_image_pixels = background.size

    reconstruction = np.stack([background] * num_frames)
    reconstruction += noises

    for i in range(num_components):
        component_slice = bounding_boxes[i].to_slices()
        cell = W[i, :].reshape(-1, 1) @ H[i, :].reshape(1, -1) + B[i, :]
        cell = cell.reshape(num_frames, *components[0].shape)
        reconstruction[(slice(None),) + component_slice] = cell

    H = H.reshape(-1, *components[0].shape)
    B = B.reshape(-1, *components[0].shape)

    component_loss = {i: float(np.linalg.norm(
                               components[i] - H[i])/num_component_pixels)
                      for i in range(num_components)}
    trace_loss = {i: float(np.linalg.norm(
                           traces[i] - W[i])/num_frames)
                      for i in range(num_components)}
    reconstruction_error = {t: float(np.linalg.norm(
                    sequence[t, :, :] - reconstruction[t, :, :])/num_image_pixels)
                    for t in range(num_frames)}

    return component_loss, trace_loss, reconstruction_error
