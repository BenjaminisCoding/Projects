"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec
import random
from tqdm import tqdm 

############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):

    ##################
    # your code here #
    ##################
    walk = [node]
    for i in range(walk_length-1):
        neighbour = list(G.neighbors(walk[i]))
        walk.append(neighbour[randint(0, len(neighbour) - 1)])
    walk = [str(node) for node in walk]
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    ##################
    # your code here #
    ##################
    for node in tqdm(G.nodes()):
        for _ in range(num_walks):
            walks.append(random_walk(G, node, walk_length))
        
    permuted_walks = np.random.permutation(walks) #to not have a structure that could be learned by a network 
    return permuted_walks.tolist()


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
