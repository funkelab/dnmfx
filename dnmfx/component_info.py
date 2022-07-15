from scipy import spatial
import math


class ComponentInfo():
    '''Represents a single component by its bounding box in the volume and a
    unique index.

    Args:

        bounding_box (:class:`funlib.geometry.Roi`):
            The bounding box of the component, in voxels.

        component_index (int):
            A unique zero-based and continuous index for each component.
    '''

    def __init__(self, bounding_box, component_index):

        self.bounding_box = bounding_box
        self.index = component_index
        self.overlapping_components = []


def create_component_info(bounding_boxes):
    '''Create a list of ComponentInfo from bounding boxes.

    Args:

        bounding_boxes (list of :class:`funlib.geometry.Roi`):
            A list of bounding boxes.

    Returns:

        A list of :class:`ComponentInfo`, where each component stores a list of
        other components it overlaps with.
    '''

    # create components from bounding boxes
    components = construct_components(bounding_boxes)

    # get all components' centers
    component_centers = [c.bounding_box.center for c in components]

    # create a KD-Tree
    kd_tree = spatial.KDTree(component_centers)

    # store overlapping components
    overlaps = {
        component.index:
        find_overlapping_components(component, components, kd_tree)
        for component in components
    }

    for index, overlapping_components in overlaps.items():
        components[index].overlapping_components = overlapping_components

    return components


def find_overlapping_components(component, components, kd_tree):

    component_center = component.bounding_box.center

    # query all components that are close to 'component'
    close_component_indices = kd_tree.query_ball_point(
        component_center,
        get_diagonal_length(component) + 1
    )
    close_components = [
        components[i]
        for i in close_component_indices
    ]

    # check if close components overlap with 'component'
    overlapping_components = []
    for close_component in close_components:

        # skip the component itself
        if close_component.index == component.index:
            continue

        if close_component.bounding_box.intersects(component.bounding_box):
            overlapping_components.append(close_component)

    return overlapping_components


def construct_components(bounding_boxes):

    return [
        Component(bounding_box, box_index)
        for box_index, bounding_box in enumerate(bounding_boxes)
    ]


def get_diagonal_length(component):

    begin = component.bounding_box.begin
    end = component.bounding_box.end

    diagonal_length_squared = sum((end - begin)**2)

    return math.sqrt(diagonal_length_squared)
