import networkx as nx
from .component_description import create_component_description


def get_groups(dataset):
    """Find all connected components in the data from component descriptions.

    Args:
        dataset (:class:`Dataset`):
            Dataset to be fitted; should contain `bounding_boxes` for the
            components.

    Returns:

        A list of lists of length the number of groups; each list contains a
        number of :class:`ComponentDescription` that form a group.
    """

    component_descriptions = \
        create_component_description(dataset.bounding_boxes)

    connection_dict = {
                component_description:
                component_description.overlapping_components
                for component_description in component_descriptions}

    # construct graph
    G = nx.Graph()
    G.add_nodes_from(connection_dict.keys())

    for component_description, overlaps in connection_dict.items():
        connections = list(zip([component_description]*len(overlaps),
                               overlaps))
        G.add_edges_from(connections)

    return [list(c) for c in nx.connected_components(G)]
