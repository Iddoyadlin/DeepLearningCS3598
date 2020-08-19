from unittest import TestCase
import networkx as nx
from scipy.spatial import distance
import numpy as np
from graph import add_nodes, get_sorted_nn_graph, get_graph


class GraphTest(TestCase):
    def setUp(self) -> None:
        self.points = [[1,1], [2,2], [1,4], [10, 10], [100,100]]

    def test_add_nodes(self):
        G = nx.Graph()
        add_nodes(G, self.points)
        pos = nx.get_node_attributes(G,'pos')
        self.assertEqual([tuple(p) for p in self.points], list(pos.values()))

    def test_get_sorted_knn_graph(self):
        distances = distance.cdist(np.array(self.points), np.array(self.points), 'euclidean')
        knn_graph = get_sorted_nn_graph(distances)
        self.assertEqual(knn_graph[0][1], 2)
        self.assertEqual(knn_graph[4][0], 3)

    def test_get_specified_k_graph(self):
        G = get_graph(self.points, k=2)
        self.assertEqual(list(G.neighbors(0)), [1,2])

    def test_get_graph_find_k(self):
        G = get_graph(self.points)
        self.assertEqual(G.degree[0], 2)

    def test_weights_are_as_expected(self):
        G = get_graph(self.points)
        actual_dist = G.get_edge_data(0,2)['weight']
        self.assertEqual(3, actual_dist)