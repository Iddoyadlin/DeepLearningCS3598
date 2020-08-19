from scipy.spatial import distance
import numpy as np
import networkx as nx

def add_edges(G, knn_graph, distances, k):
    n = G.number_of_nodes()
    deg = G.degree[0]
    for u in range(n):
        for v in range(deg, k):
            neighbor = knn_graph[u][v]
            G.add_edge(u, neighbor, weight=distances[u][neighbor])
    return G


def add_nodes(G, points):
    for u in np.arange(len(points)):
        G.add_node(u, pos=tuple(points[u]))


def get_sorted_nn_graph(distances):
    knn_graph = np.argsort(distances, axis=1, kind='stable')[:, 1:]  # node is closest to itself always
    return knn_graph


def get_graph(points, dist_type='euclidean', k=None, start_k=2, max_k=100):
    distances = distance.cdist(points, points, dist_type)
    knn_graph = get_sorted_nn_graph(distances)
    G = nx.Graph()
    add_nodes(G, points)
    if k is None:
        for i in range(start_k, max_k):
            add_edges(G, knn_graph, distances, i)
            if nx.is_connected(G):
                return G
    else:
        add_edges(G, knn_graph, distances, k)
        return G
    return None # couldnt find connection


def get_most_distant_points(points, dist_type='euclidean'):
    distances = distance.cdist(points, points, dist_type)
    I = np.unravel_index(distances.argmax(), distances.shape)
    return I
