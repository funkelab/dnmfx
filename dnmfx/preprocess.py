import numpy as np
import jax
import cv2
import random
import optax
from scipy import spatial

class Component():

    def __init__(self, bounding_box, box_index):
        # with the 2D dataset
        # bounding_box = (x_begin, x_end, y_begin, y_end)
        # begin = (x_begin, y_begin);
        # end = (x_end, y_end)
        self.begin = (bounding_box[0], bounding_box[2])
        self.end = (bounding_box[1], bounding_box[3])
        self.index = box_index


def create_overlap_lookup_dict(bounding_boxes):

    components = construct_components(bounding_boxes)
    overlaps = {
                    component.index: find_overlapping_components(component,
                                     components) for component in components
               }

    return overlaps


def find_overlapping_components(component, components):

    overlapping_components = []

    component_centers = [get_component_center(c) for c in components]
    component_center = get_component_center(component)
    tree = spatial.KDTree(component_centers)
    close_component_center_indexes = tree.query_ball_point(
                                        component_center,
                                        get_component_size(component)*2
                                     )
    close_component_centers = [component_centers[i] for i in
                               close_component_center_indexes]
    #close_component_centers.remove(component_center)
    close_components = construct_component_from_centers(close_component_centers,
                                                        components)

    for close_component in close_components:
        if overlap(close_component, component):
            overlapping_components.append(close_component)

    return overlapping_components

def overlap(componentA, componentB):
    # begin = (x_begin, y_begin);
    # end = (x_end, y_end)

    x_begin_in_range = componentA.begin[0] in range(componentB.begin[0],
                       componentB.end[0])
    x_end_in_range = componentA.end[0] in range(componentB.begin[0],
                     componentB.end[0])
    y_begin_in_range = componentA.begin[1] in range(componentB.begin[1],
                       componentB.end[1])
    y_end_in_range = componentA.end[1] in range(componentB.begin[1],
                     componentB.end[1])

    condition1 = x_begin_in_range and y_end_in_range
    condition2 = x_end_in_range and y_end_in_range
    condition3 = x_begin_in_range and y_begin_in_range
    condition4 = x_end_in_range and y_begin_in_range

    return condition1 or condition2 or condition3 or condition4


def construct_components(bounding_boxes):

    components = []
    # Take bounding boxes and convert into Component objects
    for box_index, bounding_box in enumerate(bounding_boxes):
        component = Component(bounding_box, box_index)
        components.append(component)

    return components


def get_component_center(component):

    # Take component object and find its center
    x = component.begin[0] + (component.end[0] - component.begin[0])/2
    y = component.begin[1] + (component.end[1] - component.begin[1])/2

    return (x, y)


def get_component_size(component):

    # Take component object and find its size
    width = component.end[0]-component.begin[0]
    height = component.end[1]-component.begin[1]
    # assume box is always a square
    return width


def construct_component_from_centers(component_centers, all_components):

    # Create Component objects out of component centers
    components = []
    for component_center in component_centers:
        for component in all_components:
            if get_component_center(component) == component_center:
                components.append(component)

    return components

if __name__ == "__main__":
    bounding_boxes = np.load("bounding_boxes.npy")
    overlaps = create_overlap_lookup_dict(bounding_boxes)

    for index, overlapping_components in overlaps.items():
        print(index, [c.index for c in overlapping_components])
