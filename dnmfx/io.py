from .dataset import Dataset
import funlib.geometry as fg
import numpy as np
import zarr


def read_dataset(dataset_path):
    """Read in dataset in a zarr container given dataset path.

    Args:

        dataset_path (string):
            The path to the zarr container containing the dataset. Should have
            a `sequence` dataset of shape `(t, [[z,], y,] x)` and a
            `component_locations` dataset of shape `(n, 2, d)`, where `n` is
            the number of components and `d` the number of spatial dimensions.
            `component_locations` stores the begin and end of each component,
            i.e., `component_locations[1, 0, :]` is the begin of component `1`
            and `component_locations[1, 1, :]` is its end.

    Returns:

        Dataset (:class: `Dataset`):
            Dataset that contains `bounding_boxes` as :class:
            `funlib.geometry.Roi`, components of shape `(k, s)` where `k` is
            the number of components, `s` is the component size, `traces` of
            shape `(t, k)` where `t` is the number of frames, `background` of
            shape `(image_size, image_size)`, `noises` of shape `(t,
                    image_size, image_size)`, and `sequence` dataset of shape
            `(t, [[z,], y,] x)`.
    """

    with zarr.open(dataset_path, 'r') as f:

        components = None
        traces = None
        background = None
        noises = None
        sequence = None
        if 'components' in f:
            components = f['components']
        if 'traces' in f:
            traces = f['traces']
        if 'background' in f:
            background = f['background']
        if 'noises' in f:
            noises = f['noises']
        if 'sequence' in f:
            sequence = f['sequence']

        component_locations = f['component_locations']
        bounding_boxes = [
            fg.Roi(
                fg.Coordinate(begin),
                fg.Coordinate(end) - fg.Coordinate(begin))
            for begin, end in zip(
                component_locations[:, 0, :],
                component_locations[:, 1, :]
            )
        ]

    return Dataset(
        bounding_boxes,
        components,
        traces,
        background,
        noises,
        sequence)


def write_dataset(dataset, dataset_path):
    """Write dataset to a zarr container.

    Args:

        Dataset (:class: `Dataset`):
            Dataset that contains `bounding_boxes` as :class:
            `funlib.geometry.Roi`, components of shape `(k, s)` where `k` is
            the number of components, `s` is the component size, `traces` of
            shape `(t, k)` where `t` is the number of frames, `background` of
            shape `(image_size, image_size)`, `noises` of shape `(t,
                    image_size, image_size)`, and `sequence` dataset of shape
            `(t, [[z,], y,] x)`.

        dataset_path (string):
            The path to the zarr container containing the dataset. Should have
            a `sequence` dataset of shape `(t, [[z,], y,] x)` and a
            `component_locations` dataset of shape `(n, 2, d)`, where `n` is
            the number of components and `d` the number of spatial dimensions.
            `component_locations` stores the begin and end of each component,
            i.e., `component_locations[1, 0, :]` is the begin of component `1`
            and `component_locations[1, 1, :]` is its end.
   """

    with zarr.open(dataset_path, 'a') as f:

        if dataset.components is not None:
            f['components'] = dataset.components
        if dataset.traces is not None:
            f['traces'] = dataset.traces
        if dataset.background is not None:
            f['background'] = dataset.background
        if dataset.noises is not None:
            f['noises'] = dataset.noises
        if dataset.sequence is not None:
            f['sequence'] = dataset.sequence

        component_locations = np.array([
            [roi.get_begin(), roi.get_end()]
            for roi in dataset.bounding_boxes
        ])
        f['component_locations'] = component_locations
