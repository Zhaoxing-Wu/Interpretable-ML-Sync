import networkx as nx
import matplotlib.pyplot as plt
from math import floor
import networkx as nx
import numpy as np
from NNetwork import NNetwork as nn
import pandas as pd
import pdb

num_nodes = 200
probability = 0.25
seed = 42
sample_k = 20
sample_size = 100

NWS = nx.newman_watts_strogatz_graph(num_nodes, 50, probability, seed) # type: ignore
new_nodes = {e: n for n, e in enumerate(NWS.nodes, start=1)}
new_edges = [(new_nodes[e1], new_nodes[e2]) for e1, e2 in NWS.edges]
edgelist = []
for i in range(len(new_edges)):
    temp = [str(new_edges[i][0]), str(new_edges[i][1])]
    edgelist.append(temp)
G = nn.NNetwork()
G.add_edges(edgelist)

def FCA(G, s, k, iteration):
    """Implements the Firefly Cellular Automata model
    Args:
        G (NetworkX Graph): Input graph to the model
        s (array): Current state
        k (int): k-color FCA
        iteration (int): number of iterations
    Returns:
        ret: states at each iteration
        label: whether the system concentrates at the final iteration
    """
    b = (k-1)//2  # Blinking color
    ret = s
    s_next = np.zeros(G.num_nodes())
    for h in range(iteration):
        if h != 0:
            s = s_next  # Update to the newest state
            ret = np.vstack((ret, s_next))
        s_next = np.zeros(G.num_nodes())
        for i in range(G.num_nodes()):
            flag = False  # True if inhibited by the blinking neighbor
            if s[i] > b:
                for j in range(G.num_nodes()):
#                     if s[j] == b and list(G.nodes())[j] in list(G.adj[list(G.nodes())[i]]):
                      if s[j] == b and list(G.nodes())[j] in list(G.neighbors(list(G.nodes())[i])):
                        flag = True
                if flag:
                    s_next[i] = s[i]
                else:
                    s_next[i] = (s[i]+1) % k
            else:
                s_next[i] = (s[i]+1) % k

    width = width_compute(ret[-1], k)
    label = False
    if (width < floor(k / 2)):  # half circle concentration
        label = True

    return ret, label

def width_compute(coloring, kappa):
    """Computes the width of the dynamics at current FCA coloring

    Args:
        coloring (list): List of colorings/dynamics for FCA
        kappa (int): Integer that parametrizes the FCA dynamics with $$k = kappa$$

    Returns:
        int: The width computed for FCA
    """
    differences = [np.max(coloring) - np.min(coloring)]
    for j in range(1, kappa+1):
        shifted = (np.array(coloring) + j) % kappa
        differences.append(np.max(shifted) - np.min(shifted))
    return np.min(differences)


# s = np.arccos(np.random.uniform(-1, 1, size))
s = np.random.randint(0,4,200)
dynamics, label = FCA(G, s, k=5, iteration=50)

#### Need to make this a more general width compute function
my_width = lambda x,y: width_compute([x, y], kappa=5)

def ccat(dynamics, label, sample_k, sample_size, my_width):
    """Takes in the global dynamics, its label for Sync. or Non-Sync., subgraph size or 
    the parameter k, sample size to determine how many examples to sample from our network,
    and my_width lambda function to calculate the width for dynamics

    Args:
        dynamics (_type_): _description_
        label (_type_): _description_
        sample_k (_type_): _description_
        sample_size (_type_): _description_
        my_width (_type_): _description_
    """
    X, embs = G.get_patches(k=sample_k, sample_size=sample_size, skip_folded_hom=True)
    embs = np.array(embs).astype(int)
    tensor_subgraphs = []
    for i, embed in enumerate(embs):
        embed -= 1
        final = []
        for color in dynamics:
#             pdb.set_trace()
            col_idx = color[embed]
#             pdb.set_trace()
            adj_mat = X.T[i].reshape(sample_k, sample_k) # type: ignore
            for j in range(sample_k-1):
                for k in range(j):
                    if adj_mat[j,k] > 0:
                        adj_mat[j,k] = my_width(col_idx[j], col_idx[k])
            adj_mat += adj_mat.T
            final.append(adj_mat)
        tensor_subgraphs.append((np.asarray(final), label))
    return tensor_subgraphs

# 100 parents
N = 10
subgraphs = []

for i in range(1, N):
    num_nodes = 100
    probability = 0.15
    seed = 42
    sample_k = 15
    sample_size = 3
    
    # Generates a new random parent graph
    NWS = nx.newman_watts_strogatz_graph(num_nodes, 50, probability, seed) # type: ignore
    new_nodes = {e: n for n, e in enumerate(NWS.nodes, start=1)}
    new_edges = [(new_nodes[e1], new_nodes[e2]) for e1, e2 in NWS.edges]
    edgelist = []
    for i in range(len(new_edges)):
        temp = [str(new_edges[i][0]), str(new_edges[i][1])]
        edgelist.append(temp)
    G = nn.NNetwork()
    G.add_edges(edgelist)
    
    s = np.random.randint(0,4,100)
    dynamics, label = FCA(G, s, k=5, iteration=50)
    dynamics -= 1
    
    subgraphs += ccat(dynamics, label, sample_k, sample_size, my_width)