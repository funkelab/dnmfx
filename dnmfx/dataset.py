import funlib.geometry as fg
import numpy as np


class Dataset():

    def __init__(
            self,
            bounding_boxes,
            components=None,
            traces=None,
            background=None,
            noises=None,
            sequence=None):

        self.components = components
        self.traces = traces
        self.background = background
        self.noises = noises
        self.bounding_boxes = bounding_boxes
        self.sequence = sequence
        self.num_components = len(bounding_boxes)
        self.num_spatial_dims = bounding_boxes[0].dims

        assert background is not None or sequence is not None, \
            "Either background (and components, traces, and noises) or " \
            "sequence has to be given"

        if components is not None:
            assert traces is not None, \
                "Need to provide traces if components are provided"

        if traces is not None:
            self.num_frames = traces.shape[1]
        else:
            self.num_frames = sequence.shape[0]
        if background is not None:
            self.frame_roi = fg.Roi((0,) * self.num_spatial_dims,
                                    self.background.shape)
        else:
            self.frame_roi = fg.Roi((0,) * self.num_spatial_dims,
                                    self.sequence.shape[0])

        assert traces.shape[0] == self.num_components, \
            "Traces and components disagree on number of components"
        assert noises.shape[0] == self.num_frames, \
            "Traces and noises have different numbers of frames"
        assert background.shape == noises.shape[1:], \
            "Background and noises have incompatible shapes"
        for i, bounding_box in enumerate(self.bounding_boxes):
            assert self.frame_roi.contains(bounding_box), \
                f"Bounding box {bounding_box} of component {i} does not fit " \
                "into frame"

        self.sequence_cache = {}

    def render(self, include_background=True, include_noises=True):
        """Render the sequence of this dataset from the components, traces,
        background, and noises.

        Args:

            include_background (bool):
                If set, the background component will be included.

            include_noises (bool):
                If set, pixel- and frame-wise independent additive noise will
                be added. This noise pattern is precomputed, repeated calls to
                this function will return the exact same noise patterns.
        """

        if (include_background, include_noises) in self.sequence_cache:
            return self.sequence_cache[(include_background, include_noises)]

        sequence = None

        if include_background:

            assert self.background is not None, \
                "Can't render a sequence if no background is given"

            sequence = np.stack([self.background] * self.num_frames)

        if include_noises:

            assert self.noises is not None, \
                "Can't render a sequence if no noises are given"

            if sequence is None:
                sequence = self.noises
            else:
                sequence += self.noises

        assert self.components is not None, \
            "Can't render a sequence if no components are given"

        # render each component
        for component, bounding_box, trace in zip(
                self.components,
                self.bounding_boxes,
                self.traces):

            component_sequence = np.stack([component] * self.num_frames)
            component_sequence *= trace[:, np.newaxis, np.newaxis]

            print(bounding_box)
            slices = bounding_box.to_slices()
            print(slices)
            sequence[(slice(None),) + slices] += component_sequence

        self.sequence_cache[(include_background, include_noises)] = sequence

        return sequence
