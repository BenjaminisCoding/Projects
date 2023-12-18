"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import os

############## Task 1

##################
# your code here #
##################


path_data = 'C:/Users/benja/Documents/MVA/P1/Altegrad/TP2/lab2_graph_mining/code/datasets/'

G = nx.read_edgelist(os.path.join(path_data, 'CA-HepTh.txt'), delimiter='\t')
print(G, '\n') #print the number of nodes and edges

################
# Question 1 ###
################

#n parmi 3 


############## Task 2

##################
# your code here #
##################

num_connected_components = len([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
print('Number of connected components :', num_connected_components)
if num_connected_components > 1 :
    
    largest_cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(largest_cc)

    num_nodes_largest = subG.number_of_nodes()
    num_edges_largest = subG.number_of_edges()

    print("Number of nodes in the largest connected component:", num_nodes_largest)
    print("Number of edges in the largest connected component:", num_edges_largest)

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    fraction_nodes = num_nodes_largest / total_nodes
    fraction_edges = num_edges_largest / total_edges

    print("Fraction of nodes in the largest connected component:", fraction_nodes)
    print("Fraction of edges in the largest connected component:", fraction_edges, '\n')

############## Task 3
# Degree

degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################

min_degree = min(degree_sequence)
max_degree = max(degree_sequence)
median_degree = np.median(degree_sequence)
mean_degree = np.mean(degree_sequence)

print("Minimum Degree:", min_degree)
print("Maximum Degree:", max_degree)
print("Median Degree:", median_degree)
print("Mean Degree:", mean_degree, '\n')
############## Task 4

##################
# your code here #
##################

degree_hist = nx.degree_histogram(G)


plt.figure(figsize=(8, 8))
plt.bar(range(len(degree_hist)), degree_hist, width=0.80, color='b', align='center')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Histogram')

plt.show()

# Now, let's create a log-log plot of the degree histogram
plt.figure(figsize=(8, 8))
plt.loglog(range(len(degree_hist)), degree_hist, 'b.', markersize=10)
plt.xlabel('Degree (log scale)')
plt.ylabel('Frequency (log scale)')
plt.title('Log-Log Degree Histogram')

plt.show()


############## Task 5

##################
# your code here #
##################

print('The transitivity of the graph is', nx.transitivity(G), 'NB: in the documentation, the coefficient is multiplied by 3')
