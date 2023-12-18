"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt

from code_lab_community_detection import spectral_clustering


# Loads the karate network

#### Uncomment so it works on your machine

#G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
G = nx.read_weighted_edgelist(r"C:\Users\benja\Documents\MVA\P1\Altegrad\TP5\code\data\karate.edgelist", delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
#class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
class_labels = np.loadtxt(r"C:\Users\benja\Documents\MVA\P1\Altegrad\TP5\code\data\karate_labels.txt", delimiter=',', dtype=np.int32)

idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
# your code here #
##################

node_colors = ['red' if label == 1 else 'blue' for label in y]

nx.draw_networkx(
    G,
    labels=idx_to_class_label,
    font_color='black',
    font_size=12,
    font_weight='bold',
    node_color=node_colors
)
plt.show()

############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model


n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)# your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
# your code here #
##################

clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_pred))
############## Task 8
# Generates spectral embeddings

##################
# your code here #
##################

labels = spectral_clustering(G, 2)
print("Accuracy of spectral clustering", np.max([accuracy_score(list(labels.values()), list(idx_to_class_label.values())), 1 - accuracy_score(list(labels.values()), list(idx_to_class_label.values()))]))
#we use the max function to obtain the correct accuracy as the spetral clustering algorithm can choose to label 0 the class 1 in our dataset