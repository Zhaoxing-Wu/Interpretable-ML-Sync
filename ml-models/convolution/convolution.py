# Large Graphs -> n = 100, 200
# Run dynamics -> Get ground truth

# MCMC Sampling -> n = 10, 15, 20
# Also sample the dynamics that were originally run on these nodes that we sampled
# Recursively average the probabilities to get the final label for this large graph

import networkx as nx
import numpy as np
from NNetwork import NNetwork as nn
import pandas as pd
import copy

def generate_nxg(X):
    graph_list = []
    k = int(np.sqrt(X.shape[0]))
    for i in range(X.shape[1]):
        adj_mat = X.T[i].reshape(k,k)
        G = nx.from_numpy_matrix(adj_mat)
        graph_list.append(G)
    
    return graph_list

def Kuramoto(G, s, iteration, step):
    ret = s
    s_next = np.zeros(G.num_nodes())
    for h in range(iteration-1):
        G_ind = copy.deepcopy(G)
        if h != 0:
            s = s_next  # Update to the newest state
            ret = np.vstack((ret, s_next))
        for i in range(G.num_nodes()):
            neighbor_col = []
            for j in range(G.num_nodes()):
                if list(G.nodes())[j] in list(G.adj[list(G.nodes())[i]]):
                    neighbor_col.append(s[j])

            new_col = s[i] + step * np.sum(np.sin(neighbor_col - s[i]))
            if np.abs(new_col) > np.pi:
                if new_col > np.pi:
                    new_col -= 2*np.pi
                if new_col < -np.pi:
                    new_col += 2*np.pi
            s_next[i] = new_col

    label = False
    if widthkura(ret[-1]) < np.pi:
        label = True

    return ret, label

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
                    if s[j] == b and list(G.nodes)[j] in list(G.adj[list(G.nodes)[i]]):
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



def widthkura(colors):
    """
    computes kuramoto width from a color list
    from L2Psync repo
    """
    ordered = list(np.pi - colors)
    ordered.sort()
    lordered = len(ordered)
    threshold = np.pi

    if lordered == 1:
        return 0

    elif lordered == 2:
        dw = ordered[1]-ordered[0]
        if dw > threshold:
            return 2*np.pi - dw
        else:
            return dw

    else:
        widths = [2*np.pi+ordered[0]-ordered[-1]]
        for i in range(lordered-1):
            widths.append(ordered[i+1]-ordered[i])
        return np.abs(2*np.pi - max(widths))


def gen_large_nws(num_nodes, probability, seed, sample_k, sample_size):
    
    # Generate large NWS graph G using varying parameters
    NWS = nx.newman_watts_strogatz_graph(num_nodes, 50, probability, seed)
    new_nodes = {e: n for n, e in enumerate(NWS.nodes, start=1)}
    new_edges = [(new_nodes[e1], new_nodes[e2]) for e1, e2 in NWS.edges]
    edgelist = []
    for i in range(len(new_edges)):
        temp = [str(new_edges[i][0]), str(new_edges[i][1])]
        edgelist.append(temp)
    G = nn.NNetwork()
    G.add_edges(edgelist)
    
    # Run dynamics on this large NWS graph
    size = G.num_nodes()
    # s = np.arccos(np.random.uniform(-1, 1, size))
    s = np.random.randint(0,4,5)
    dynamics, label = FCA(G, s, k=5, iteration=50)

    # Get patches
    sampling_alg = 'pivot'
    ntwk = 'NWS'
    ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])

    # Get patches and then store as networkx graphs in a list
    X, embs = G.get_patches(k=sample_k, sample_size=sample_size, skip_folded_hom=True)
    graph_list = generate_nxg(X)

