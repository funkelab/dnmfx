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

    pairings, loss_matrix = pair(components, bounding_boxes, background, H, B, k)
    mispairings = [(c, c_hat) for c, c_hat in pairings if c != c_hat]

    component_loss = {i: loss_matrix[i , j] for i, j in pairings[:k]}
    trace_loss = {i: np.linalg.norm(traces[i, :] - W[j, :])
                  for i, j in pairings[:k]}

    return mispairings, component_loss, trace_loss


def pair(components,
         bounding_boxes,
         background,
         H, B, k):

    """
    Pair each decomposed component and component background with its corresponding
    ground truth by constructing a loss matrix, in which every element (e.g. x)
    represents the L2 norm distance between the factorization results and ground
    truth.

    More specifically, the loss matrix is structured as follows (here `C` stands for
    Component; `B` stands for component Background):

            C_0   C_1   ...   C_k   B_0   B_1   ...     B_k
    Ĉ_0  |   x                                           |
    Ĉ_1  |                                               |
    .    |                                               |
    .    |                                               |
    .    |                                               |
    Ĉ_k  |                                               |
    B̂_0  |                                               |
    B̂_1  |                                               |
    .    |                                               |
    .    |                                               |
    .    |                                               |
    B̂_k  |                                               |

    Args:

        components (list of :class:`ComponentDescription`):
            Each component stores a list of other components it overlaps with.

        bounding_boxes (:class:`funlib.geometry.Roi`):
            A list of bounding boxes of the components.

        background (array-like, shape: `(image_size, image_size)`):
            The background image in assembling the synthetic dataset.

        H (array-like, shape `(k, w*h)`):

            The factorized matrix that contains all decomposed components.

        B (array-like, shape `(k, w*h)`):

            The factorized matrix that contains the backgrounds of all decomposed
            components.

        W (array-like, shape `(t, k)`):

            The factorized matrix that contains the traces of all decomposed
            components.
    """

    loss_matrix = np.zeros((k*2, k*2))

    for i in range(k*2):
        for j in range(k*2):
            if i < k:
                component_a = H[i, :, :]
            elif i >= k:
                component_a = B[i-k, :, :]
                component_slice = bounding_boxes[i-k].to_slices()
            if j < k:
                component_b = components[j, :, :]
            elif j >= k:
                component_slice = bounding_boxes[j-k].to_slices()
                component_b = background[component_slice]
            loss_matrix[i][j] = np.linalg.norm(component_a - component_b)

    ground_truth_indices, reconstruction_indices = linear_sum_assignment(loss_matrix)
    pairings = list(zip(ground_truth_indices, reconstruction_indices))

    return pairings, loss_matrix
