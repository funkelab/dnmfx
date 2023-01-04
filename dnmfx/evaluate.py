import numpy as np
from .dataset import Dataset


def evaluate(H, W, B, dataset, show_diff=False):
    """Get the number of component and component background ID mismatches and
    find the reconstruction error per component and per trace.

    Args:

        H (array-like, shape `(k, w*h)`):
            The factorized matrix that contains all decomposed components.

        W (array-like, shape `(k, t)`):
            The factorized matrix that contains the traces of all decomposed
            components.

        B (array-like, shape `(k, w*h)`):
            The factorized matrix that contains the background of decomposed
            components.

        dataset (:class: `Dataset`):
            Dataset to be fitted; should have a `sequence` array of shape `(t,
            [[z,], y,] x)` and `bounding_boxes` of the components.

    Returns:

        reconstruction_error (float):
            The L2 distance between the input data and its reconstruction
            average by the number of pixels.

        diff_sequence (:class: `numpy.array`):
            The difference between input data and its reconstruction as a
            sequence of shape `(num_frames, image_size, image_size)`.
    """

    num_components = dataset.num_components
    num_frames = dataset.num_frames
    bounding_boxes = dataset.bounding_boxes
    ground_truth_sequence = dataset.sequence

    reconstructed_dataset = reconstruct_dataset(dataset, H, W, B)
    reconstruction = reconstructed_dataset.render(include_noises=False)

    mask = np.zeros(shape=(num_frames, *dataset.background.shape))

    for index in range(num_components):
        component_slices = bounding_boxes[index].to_slices()
        mask[(slice(None),) + component_slices] = 1

    diff_sequence = (reconstruction - ground_truth_sequence) * mask
    loss = np.linalg.norm(diff_sequence) / np.count_nonzero(mask == 1)

    if show_diff:
        return loss, diff_sequence
    else:
        return loss


def reconstruct_dataset(dataset, H, W, B):
    """Reconstruct input data from optimization results.

    Args:

        dataset (:class: `Dataset`):
            Dataset to be fitted; should have a `sequence` array of shape `(t,
            [[z,], y,] x)` and a `component_locations` array of shape `(n, 2,
            d)`, where `n` is the number of components and `d` the number of
            spatial dimensions. `component_locations` stores the begin and end
            of each component, i.e., `component_locations[1, 0, :]` is the
            begin of component `1` and `component_locations[1, 1, :]` is its
            end.

        H (array-like, shape `(k, w*h)`):
            The factorized matrix that contains all decomposed components.

        W (array-like, shape `(k, t)`):
            The factorized matrix that contains the traces of all decomposed
            components.

        B (array-like, shape `(k, w*h)`):
            The factorized matrix that contains the background of decomposed
            components.

    Returns:

        Reconstructed dataset as :class: `Dataset`
    """

    bounding_boxes = dataset.bounding_boxes
    ground_truth_components = dataset.components
    ground_truth_traces = dataset.traces
    ground_truth_background = dataset.background
    num_components = dataset.num_components

    # create a reconstruction dataset
    background = np.zeros_like(ground_truth_background)
    components = np.zeros_like(ground_truth_components)
    traces = np.zeros_like(ground_truth_traces)

    for i in range(num_components):

        component_slice = bounding_boxes[i].to_slices()
        component_shape = bounding_boxes[i].get_shape()

        # add estimated background
        background[component_slice] += B[i].reshape(component_shape)

        # add component
        components[i] = H[i].reshape(component_shape)

        # add traces
        traces[i] = W[i]

    return Dataset(
            bounding_boxes,
            background=background,
            components=components,
            traces=traces)
