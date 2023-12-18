"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4
        
##################
# your code here #
##################

Gs = [nx.cycle_graph(i) for i in range(10,20)]

############## Task 5
        
##################
# your code here #
##################

adj = sp.block_diag([nx.adjacency_matrix(G) for G in Gs])
X = np.ones((adj.shape[0], 1))
idx = []
for i,G in enumerate(Gs):
    idx += [i]*G.number_of_nodes()

adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
X = torch.FloatTensor(X).to(device)
idx = torch.LongTensor(idx).to(device)

#print(idx)


############## Task 8
        
##################
# your code here #
##################


model = GNN(1, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
print(f'vector representation neighbor_aggrfor={neighbor_aggr} and readout={readout}',model(X, adj, idx),'\n')
model = GNN(1, hidden_dim, output_dim, neighbor_aggr = 'sum', readout = 'sum', dropout = 0).to(device)
print(f'vector representation neighbor_aggrfor=sum and readout=sum',model(X, adj, idx),'\n')
model = GNN(1, hidden_dim, output_dim, neighbor_aggr = 'mean', readout = 'sum', dropout = 0).to(device)
print(f'vector representation neighbor_aggrfor=mean and readout=sum',model(X, adj, idx),'\n')
model = GNN(1, hidden_dim, output_dim, neighbor_aggr = 'sum', readout = 'mean', dropout = 0).to(device)
print(f'vector representation neighbor_aggrfor=sum and readout=mean',model(X, adj, idx),'\n')


############## Task 9
        
##################
# your code here #
##################

# Create an undirected graph
G1 = nx.Graph()
G1.add_nodes_from([k for k in range(1,7)])
G1.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)])

G2 = nx.cycle_graph(6)

plt.subplot(1, 2, 1)
nx.draw(G1, with_labels=True, font_weight='bold')
plt.title('Graph 1')

plt.subplot(1, 2, 2)
nx.draw(G2, with_labels=True, font_weight='bold')
plt.title('Graph 2')

plt.show()

print()
############## Task 10
        
##################
# your code here #
##################
Gs = [G1,G2]
adj = sp.block_diag([nx.adjacency_matrix(G) for G in Gs])
X = np.ones((adj.shape[0], 1))
idx = []
for i,G in enumerate(Gs):
    idx += [i]*G.number_of_nodes()

adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
X = torch.FloatTensor(X).to(device)
idx = torch.LongTensor(idx).to(device)


############## Task 11
        
##################
# your code here #
##################

model = GNN(1, hidden_dim, output_dim, neighbor_aggr = 'sum', readout = 'sum', dropout = 0).to(device)
print(f'vector representation neighbor_aggrfor=sum and readout=sum',model(X, adj, idx),'\n')

############## Question 4


G1 = nx.Graph()
G1.add_nodes_from([k for k in range(1,9)])
G1.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5)])

G2 = nx.cycle_graph(8)
print(np.sum(nx.adjacency_matrix(G1).todense(), axis=1))
print(np.sum(nx.adjacency_matrix(G2).todense(), axis=1))

plt.subplot(1, 2, 1)
nx.draw(G1, with_labels=True, font_weight='bold')
plt.title('Graph 1')

plt.subplot(1, 2, 2)
nx.draw(G2, with_labels=True, font_weight='bold')
plt.title('Graph 2')

plt.show()

Gs = [G1,G2]
adj = sp.block_diag([nx.adjacency_matrix(G) for G in Gs])
X = np.ones((adj.shape[0], 1))
idx = []
for i,G in enumerate(Gs):
    idx += [i]*G.number_of_nodes()

adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
X = torch.FloatTensor(X).to(device)
idx = torch.LongTensor(idx).to(device)

model = GNN(1, hidden_dim, output_dim, neighbor_aggr = 'sum', readout = 'sum', dropout = 0).to(device)
print(f'vector representation neighbor_aggrfor=sum and readout=sum',model(X, adj, idx),'\n')