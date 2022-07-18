


def get_groups(component_description):
    sorted_component_description = sorted(component_description, key=lambda x:
            len(x.overlapping_components))
    return run_depth_first_search(sorted_component_description,
                                  sorted_component_description,
                                  sorted_component_description[0],
                                  [],
                                  [])

def run_depth_first_search(unvisited,
                           component_description,
                           component,
                           group,
                           groups):
    index = component.index
    if component in unvisited:
        unvisited.remove(component)
        overlaps = component.overlapping_components

        if len(overlaps) == 0:
            groups.append([component])
        else:
            all_overlaps_added = True
            for overlap in overlaps:
                if overlap not in group:
                    all_overlaps_added = False
                    group.append(overlap)
                    run_depth_first_search(unvisited,
                                           component_description,
                                           overlap,
                                           group,
                                           groups)
            if all_overlaps_added:
                groups.append(group)
                if len(unvisited) != 0:
                    run_depth_first_search(unvisited,
                                           component_description,
                                           unvisited[0],
                                           [],
                                           groups)
    elif len(unvisited) != 0:
        groups.append(group)
        run_depth_first_search(unvisited,
                               component_description,
                               unvisited[0],
                               [],
                               groups)

    return groups


def get_total_intersect(component, frame):

    total_intersect = jnp.zeros(component.bounding_box.shape, dtype=jnp.float32)
    for overlapping_component in component.overlapping_components:
        total_intersect.at[:, :].add(get_intersect(component.bounding_box,
                                     overlapping_component.bounding_box,
                                     frame))

    return total_intersect.flatten()


def get_intersect(bounding_box_A, bounding_box_B, frame):

    intersect = jnp.zeros(bounding_box_A.shape, dtype=jnp.float32)
    intersect_roi = bounding_box_A.intersect(bounding_box_B)

    row_shift = bounding_box_A.get_end()[1]
    col_shift = bounding_box_A.get_begin()[0]

    start_col, start_row = intersect_roi.get_begin()
    end_col, end_row = intersect_roi.get_end()

    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            intersect.at[row-row_shift, col-col_shift].set(frame[row, col])

    return intersect
