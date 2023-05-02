import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

class Graph:
    def __init__(self, image):
        self.image = image
        self.width, self.height = image.shape[1], image.shape[0]

    def construct(self, neighborhood_size=10):
        xs, ys = np.where(self.image == 255)
        points = np.column_stack((xs, ys))

        kdtree = cKDTree(points)
        edges = kdtree.query_pairs(neighborhood_size)

        graph = nx.Graph()
        graph.add_nodes_from(range(len(points)))
        graph.add_edges_from(edges)

        return graph

    def get_shortest_path(self, start, end):
        graph = self.construct()
        path = nx.shortest_path(graph, start, end)

        return path

    def get_shortest_path_length(self, start, end):
        graph = self.construct()
        path_length = nx.shortest_path_length(graph, start, end)

        return path_length
