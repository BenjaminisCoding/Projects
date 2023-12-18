"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans
import random
import os

path_data = 'C:/Users/benja/Documents/MVA/P1/Altegrad/TP2/lab2_graph_mining/code/datasets/'
G = nx.read_edgelist(os.path.join(path_data, 'CA-HepTh.txt'), delimiter='\t')

############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):

    ##################
    A = nx.adjacency_matrix(G)
    D_inv = diags([1/G.degree(node) for node in G.nodes()])
    laplacian_matrix = eye(G.number_of_nodes()) - D_inv @ A
    _, eigenvectors = eigs(laplacian_matrix, k=k, which='SR')
    eigenvectors = np.real(eigenvectors)
    kmeans = KMeans(n_clusters=k).fit(eigenvectors)
    clustering = {}
    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]
    ##################

    return clustering


############## Task 7

##################
# your code here #
##################

largest_cc = max(nx.connected_components(G))
subG = G.subgraph(largest_cc)
clustering_of_largest_cc = spectral_clustering(subG, 50)

############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    
    modularity = 0
    m = nx.number_of_edges(G)
    edges_inside_community = {}
    sum_degrees_inside_community = {}

    for community in set(clustering.values()):

        nodes_in_community = [node for node, comm in clustering.items() if comm == community]
        subgraph = G.subgraph(nodes_in_community)
        edges_count = nx.number_of_edges(subgraph)
        degrees_sum = sum(dict(subgraph.degree()).values())

        edges_inside_community[community] = edges_count
        sum_degrees_inside_community[community] = degrees_sum
        modularity += (edges_count / m) - (degrees_sum / (2 * m)) ** 2

    return modularity



############## Task 9

##################
# your code here #
##################

def random_clustering(G, k):

    ##################
    clustering = {}
    for i, node in enumerate(G.nodes()):
        clustering[node] = random.randint(0,k)
    ##################
    return clustering

random_clustering = random_clustering(subG, 50)
print('Modularity for a random_clustering is:', modularity(subG, random_clustering))
print('Modularity for our clustering is:', modularity(subG, clustering_of_largest_cc))




