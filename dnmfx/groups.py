import networkx as nx
import pickle
from .component_description import create_component_description
from .io import read_dataset


def get_groups(dataset_path, file_to_save):
    """Find all connected components in the data from component descriptions.

     Args:

        component_descriptions (list of :class:`ComponentDescription`):

            The bounding boxes and indices of the components to estimate.

    Returns:

        A list of lists, each of which is a list of :class:`ComponentDescription`
        that are connected.
    """
    dataset = read_dataset(dataset_path)
    component_descriptions = create_component_description(dataset.bounding_boxes)

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

    groups = [list(c) for c in nx.connected_components(G)]
    with open(file_to_save, "wb") as f:
        pickle.dump(groups, f)
