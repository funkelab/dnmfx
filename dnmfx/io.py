from .dataset import Dataset
import funlib.geometry as fg
import numpy as np
import zarr


def read_dataset(dataset_path):

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
