import networkx as nx


def get_groups(component_descriptions):
    """Find all connected components in the data from component descriptions.

     Args:

        component_descriptions (list of :class:`ComponentDescription`):

            The bounding boxes and indices of the components to estimate.

    Returns:

        A list of lists, each of which is a list of :class:`ComponentDescription`
        that are connected.
    """

    connection_dict = {
                component_description:
                component_description.overlapping_components
                for component_description in component_descriptions}

    # construct graph
    G = nx.Graph()
    G.add_nodes_from(connection_dict.keys())

    for component_description, overlaps in connection_dict.items():
        connections = list(zip([component_description]*len(overlaps), overlaps))
        G.add_edges_from(connections)

    return [list(c) for c in nx.connected_components(G)]
