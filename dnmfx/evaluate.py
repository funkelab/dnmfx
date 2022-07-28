from dnmfx.io import read_dataset
import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate(H, B, W, dataset):

    """
    Get the number of component and component background ID mismatches and
    find the reconstruction error per component and per trace.

    Args:

        H (array-like, shape `(k, w*h)`):

            The factorized matrix that contains all decomposed components.

        B (array-like, shape `(k, w*h)`):

            The factorized matrix that contains the backgrounds of all decomposed
            components.

        W (array-like, shape `(t, k)`):

            The factorized matrix that contains the traces of all decomposed
            components.

        dataset (:class: `Dataset`):

            Dataset to be factorized.
    """

    components = dataset.components
    background = dataset.background
    traces = dataset.traces
    bounding_boxes = dataset.bounding_boxes
    k = dataset.num_components

    H = H.reshape(-1, *components[0].shape)
    B = B.reshape(-1, *components[0].shape)
    W = W.reshape(traces.shape)

    loss_matrix = np.zeros((k*2, k*2))
    for i in range(k*2):
        for j in range(k*2):
            if i < k:
                component_a = components[i, :, :]
            elif i >= k:
                component_slice = bounding_boxes[i-k].to_slices()
                component_a = background[component_slice]
            if j < k:
                component_b = H[j, :, :]
            elif j >= k:
                component_b = B[j-k, :, :]
            loss_matrix[i][j] = np.linalg.norm(component_a - component_b)

    ground_truth_indices, reconstruction_indices = linear_sum_assignment(loss_matrix)
    matches = list(zip(ground_truth_indices, reconstruction_indices))
    component_reconstruction_loss = \
            loss_matrix[ground_truth_indices, reconstruction_indices].sum()

    trace_reconstruction_loss = 0
    for i, j in matches[:k]:
        trace_reconstruction_loss += np.linalg.norm(traces[i, :]-W[j, :])

    return matches, component_reconstruction_loss, trace_reconstruction_loss
