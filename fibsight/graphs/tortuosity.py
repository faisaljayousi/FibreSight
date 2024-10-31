from .attributes_factory import GraphAttribute
import numpy as np

class Tortuosity(GraphAttribute):
    def __call__(self, G, metric="l2"):
        return tortuosity(G, metric)


def tortuosity(G):
    tortuosity_dict = {}

    for (start_node, end_node, edge_data) in G.edges.data("details"):
        x0, y0 = map(int, start_node.split("-"))
        x1, y1 = map(int, end_node.split("-"))

        segment_length = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

        if segment_length == 0.0:
            tortuosity = np.nan
        else:
            curve_length = edge_data.measures.length
            tortuosity = curve_length / segment_length

        # edge_data._cache["features"] = {
        #     "start": start_node,
        #     "end": end_node,
        #     "tortuosity": tortuosity,
        # }

        tortuosity_dict[(start_node, end_node)] = tortuosity

    return tortuosity_dict
