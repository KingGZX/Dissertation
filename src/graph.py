import numpy as np
from loadata import Config


class Graph:
    """
    build a human body graph
    """

    def __init__(self, max_hop=1):
        self.adjacency = None
        self.center = None
        self.num_node = None
        self.max_hop = max_hop
        self.get_edges()
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency()

    def get_edges(self):
        self.num_node = len(Config.nodes)
        self_link = [(i, i) for i in range(self.num_node)]
        # pelvis -> neck, upper leg   neck -> head, shoulder
        neighbor_link = [(0, 1), (0, 11), (0, 15), (1, 2), (1, 3), (1, 7),
                         # shoulder -> upper arm  upper arm ->forearm
                         (3, 4), (4, 5), (5, 6), (7, 8), (8, 9), (9, 10),
                         # upper leg -> lower leg   lower leg -> foot   foot -> toe
                         (11, 12), (12, 13), (13, 14), (15, 16), (16, 17), (17, 18)]
        self.edge = self_link + neighbor_link
        self.center = 0  # use pelvis as the center of body

    def get_adjacency(self):
        """
        use uniform partition to have a test first:
        which means, as long as there's path between 2 nodes and the distance is valid
        then the graph has an edge 1.
        """
        self.adjacency = np.zeros((1, self.num_node, self.num_node))
        self.adjacency[0][self.hop_dis <= self.max_hop] = 1
        # normalization
        self.adjacency[0] = undirected_graph_norm(self.adjacency[0])


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    """
    copied from st-gcn source code:
    link: https://github.com/yysijie/st-gcn
    
    My naive thinking of transfer matrix:
    when d = 0, it's identity matrix and it's simple to explain that the root node to the node itself has distance 0
    when d = 1, it's just adjacency matrix.
    when d >= 2, use the concept of matrix multiplication row map, e.g., AA = C, 
    which means, the first row of C is just: sum(A[0,i] * A[i]) ,
    therefore we can update the distance of node1 to the other nodes especially 
    those node1 do not directly link to but node1's neighbor link to.
    
    by multiply again and again, we can explore all the nodes that one node can go to.
    """
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def undirected_graph_norm(adjacency):
    """
    :param adjacency:
        adjacency matrix of the graph
    :return:
        xxx after normalization

    Note: Graph Convolution Formula
    f = (D^-1/2 A D^-1/2 H W)
    """
    degree = np.sum(adjacency, axis=1)
    degree = np.sqrt(degree)
    deg_matrix = np.multiply(np.identity(len(degree)), degree)
    adjacency = np.dot(np.dot(deg_matrix, adjacency), adjacency)
    return adjacency

# code for debugging
# ins = Graph()
# b = 0
