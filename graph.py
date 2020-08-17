from scipy.spatial import distance
import numpy as np
import networkx as nx

def get_minimal_connected_graph(points, dist_type='euclidean', start_from=2):
    for i in range(start_from, len(points)):
        G = get_graph(points, i, dist_type)
        if nx.is_connected(G):
            return G

def get_graph(points, k, dist_type='euclidean'):
    distances = distance.cdist(points, points, dist_type)
    knn_graph = np.argsort(distances, axis=1)[:, 1:k + 1]  # node is closest to itself always
    weights = np.sort(distances, axis=1)[:, 1:k + 1]  # node is closest to itself always

    G = nx.Graph()
    for u in np.arange(len(points)):
        G.add_node(u, pos=tuple(points[u]))

    for u in np.arange(len(points)):
        for j in np.arange(k):
            G.add_edge(u, knn_graph[u][j], weight=weights[u][j])
    return G

def get_most_distant_points(points, dist_type='euclidean'):
    distances = distance.cdist(points, points, dist_type)
    I = np.unravel_index(distances.argmax(), distances.shape)
    return I