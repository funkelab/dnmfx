from .groups import get_groups
from .initialize import initialize_normal
from .optimize import dnmf
from .parameters import Parameters
from datetime import datetime
import jax.numpy as jnp


def fit(
        dataset,
        max_iteration=10000,
        min_loss=1e-4,
        batch_frames=16,
        batch_components=32,
        step_size=1e-1,
        l1_weight=0,
        log_every=100,
        log_gradients=False,
        random_seed=None):

    """
    Use distributed NMF to estimate the components and traces for a dataset
    given as a zarr container.

    Args:

        dataset (:class:`Dataset`):
            Dataset to be fitted; should have a `sequence` array of shape `(t,
            [[z,], y,] x)` and `bounding_boxes` of the components.

        max_iteration (int):
            The maximum number of iterations to optimize for.

        min_loss (float):
            The loss value at which to stop the optimization.

        batch_frames (int):
            The number of frames to consider at once for each component during
            the stochastic gradient estimation.

        batch_components (int):
            The number of components to consider at once during the stochastic
            gradient estimation.

        step_size (float):
            The size of the gradient updates.

        l1_weight (float):
            The influence of the L1 regularizer on the components and traces.

        log_every (int):
            How often to print iteration statistics.

        log_gradients (bool):
            Whether to record gradients and factor matrices (i.e. H, B, W)
            after the 1st iteration.

        random_seed (int):
            A random seed for the initialization of `H` and `W`. If not given,
            a different random seed will be used each time.

    Returns:

        The optimization result of the dataset (i.e. H, W, B) and
        the losses stored as :class: `Log`.
    """

    if random_seed is None:
        random_seed = int(datetime.now().strftime("%Y%m%d%H%M%S"))

    parameters = Parameters(max_iteration,
                            min_loss,
                            batch_frames,
                            batch_components,
                            step_size,
                            l1_weight,
                            log_every,
                            log_gradients,
                            random_seed)

    H_groups = []
    W_groups = []
    B_groups = []
    log_groups = []

    groups = get_groups(dataset)
    component_group_index_pairings = \
        {component.index: group_index
            for group_index, group in enumerate(groups)
            for component in group}

    for group in groups:
        H_group, W_group, B_group, log_group = fit_group(
                                                    group,
                                                    dataset,
                                                    parameters)
        H_groups.append(H_group)
        W_groups.append(W_group)
        B_groups.append(B_group)
        log_groups.append(log_group)

    H, W, B = assemble(component_group_index_pairings,
                       H_groups, B_groups, W_groups)

    return H, W, B, log_groups


def fit_group(component_descriptions,
              dataset,
              parameters):
    """Use NMF to estimate the components and traces for a group from the
    dataset given as a list of :class: `ComponentDescription`

    Args:

        component_descriptions (list of :class: `ComponentDescription`):
            A list of :class: `ComponentDescription` that form a group in the
            dataset.

        dataset_path (string):
            The path to the zarr container containing the dataset. Should have
            a `sequence` dataset of shape `(t, [[z,], y,] x)` and
            `bounding_boxes` of the components.

        parameters (:class: `Parameters`):
            Parameters to control the optimization.

    Returns:

        The optimization result of a group from the dataset
        (i.e. `H_group`, `W_group`, `B_group`) and the losses stored as
        :class: `Log`.
    """

    num_components = len(component_descriptions)
    component_size = None
    component_shape = None
    max_overlaps = 0
    for description in component_descriptions:
        size = description.bounding_box.get_size()
        if component_size is not None:
            assert component_size == size, \
                    "Only components of the same size are supported for now"
        else:
            component_size = size
            component_shape = description.bounding_box.shape

        overlaps = len(description.overlapping_components)
        max_overlaps = max(overlaps, max_overlaps)

    print(f"Found {num_components} components of size {component_shape}")
    print(f"Maximum number of overlapping components: {max_overlaps}")

    print("Creating H index maps for each component...")
    H_indices = jnp.arange(component_size).reshape(*component_shape)
    for description in component_descriptions:
        H_index_map, W_index_map = create_index_maps(
            description,
            max_overlaps,
            H_indices)
        description.H_index_map = H_index_map
        description.W_index_map = W_index_map
    print("...done.")

    sequence = dataset.sequence
    num_frames = sequence.shape[0]
    num_components = dataset.num_components

    H_logits, W_logits, B_logits = initialize_normal(
            num_components,
            num_frames,
            component_size,
            parameters.random_seed)

    H, W, B, log = dnmf(
            dataset,
            component_descriptions,
            parameters,
            H_logits,
            W_logits,
            B_logits)

    return H, W, B, log


def assemble(component_group_index_pairings,
             H_groups,
             B_groups,
             W_groups):
    """Assemble optimization results from all groups in the dataset.

    Args:

        component_group_index_pairings (dictionary):
            A dictionary of size number of the components that
            map component index (key) to group index (value).

        H_groups (list):
            A list of `H_group` obtained from optimization result
            of a single group in the dataset.

        B_groups (list):
            A list of `B_group` obtained from optimization result
            of a single group in the dataset.

        W_groups (list):
            A list of `W_group` obtained from optimization result
            of a single group in the dataset.

    Returns:
         The optimization result of the dataset (i.e. H, W, B).
    """

    num_components = len(component_group_index_pairings)
    group_index = component_group_index_pairings[0]

    H = H_groups[group_index][0]
    B = B_groups[group_index][0]
    W = W_groups[group_index][0]

    for component_index in range(1, num_components):
        group_index = component_group_index_pairings[component_index]
        H = jnp.vstack((H, H_groups[group_index][component_index]))
        B = jnp.vstack((B, B_groups[group_index][component_index]))
        W = jnp.vstack((W, W_groups[group_index][component_index]))

    return H, W, B


def create_index_maps(component_description, max_overlaps, H_indices):

    NO_VALUE = jnp.iinfo(jnp.int32).max

    bb_i = component_description.bounding_box

    # default indices map to NO_VALUE
    H_index_map = jnp.ones((max_overlaps + 1,) + bb_i.shape, dtype=jnp.int32)
    H_index_map = H_index_map * NO_VALUE

    W_index_map = [0] * (max_overlaps + 1)

    all_components = \
        [component_description] + \
        component_description.overlapping_components

    for c, component in enumerate(all_components):

        j = component.index
        bb_j = component.bounding_box

        intersection = bb_i.intersect(bb_j)
        intersection_in_c_i = intersection - bb_i.get_begin()
        intersection_in_c_j = intersection - bb_j.get_begin()

        slices_i = intersection_in_c_i.to_slices()
        slices_j = intersection_in_c_j.to_slices()

        indices_j = H_indices[slices_j]
        H_index_map = H_index_map.at[(c,) + slices_i].set(indices_j)
        W_index_map[c] = j

    H_index_map = H_index_map.reshape(max_overlaps + 1, -1)
    W_index_map = jnp.array(W_index_map, dtype=jnp.int32)

    return H_index_map, W_index_map
