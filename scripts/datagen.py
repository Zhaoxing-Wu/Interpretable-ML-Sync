
############# Importing Libraries ###########

from plotting_scripts.network_plots import *

import pickle, csv

import numpy as np
import pandas as pd
import statistics as s
from math import floor

import networkx as nx
from NNetwork import NNetwork as nn
# from NNetwork.NNetwork import NNetwork
from karateclub import Graph2Vec, Node2Vec

import warnings
warnings.filterwarnings("ignore")

############## Subgraph Sampling ##############

def subgraphs_realworld(ntwk, sample_size, k, filename, plot_embeds=False):
    """Generate subgraphs by sampling from the provided real-world network. Plot a sample if specified.

    Args:
        ntwk (string): Input real-world network to the model
        sample_size (int): number of subgraphs to sample
        k (int): number of nodes in each subgraph
        filename (str): output filename
        plot_embeds (bool): whether to plot the embeddings samples of the subgraphs

    Returns:
        None
    """
    ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
    path = "../Networks_all_NDL/" + str(ntwk) + '.txt'
    G = nn.NNetwork()
    G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
    X, embs = G.get_patches(k=k, sample_size=sample_size, skip_folded_hom=True)
    pickle.dump(X, open(filename, 'wb'))

    if plot_embeds:
        display_graphs(title='Induced subgraphs on {}-walks in {}'.format(k, ntwk_nonumber),
                 save_path=None, 
                 data = [X, embs],
                 grid_shape = [5, 15],
                 fig_size = [15, 5],
                 show_importance=False)

######################################################


###############################FCA##########################################################################################
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
    b = (k-1)//2 # Blinking color
    ret = s
    s_next = np.zeros(G.number_of_nodes())
    for h in range(iteration):
        if h != 0:
            s = s_next # Update to the newest state
            ret = np.vstack((ret, s_next))
        s_next = np.zeros(G.number_of_nodes())
        for i in range(G.number_of_nodes()):
            flag = False # True if inhibited by the blinking neighbor
            if s[i] > b:
                for j in range(G.number_of_nodes()):
                    if s[j] == b and list(G.nodes)[j] in list(G.adj[list(G.nodes)[i]]):
                        flag = True
                if flag:
                    s_next[i] = s[i]
                else:
                    s_next[i] = (s[i]+1)%k
            else:
                s_next[i] = (s[i]+1)%k
    
    width = width_compute(ret[-1], k)
    label = False
    if (width < floor(k / 2)):  # half circle concentration
        label = True
        
    return ret, label

def datagen_FCA_dynamics(num_nodes, kappa, pred_iter, train_iter, file_name, X):
    """Generate FCA dynamics dataset

    Args:
        num_nodes: number of nodes of underlying graphs
        kappa: k-color FCA
        pred_iter: prediction iteration (label)
        train_iter: training iteration (baseline)
        file_name: output filename
        X: a list of vectorized adjacency matrix (for each graph, length = k^2 if num_nodes=k)

    Output:
        a csv file with label y, baseline_width
        si_j: the coloring of node j at time i
        q1_si, q2_si, q3_si: the 1st, 2nd, 3rd quartile at time i
        width_si: the width of time i
    """
    file = open(file_name, 'w+', newline='')
    header = ["y", "baseline_width"]
    for i in range(train_iter):
        for j in range(num_nodes):
            header.append("s" + str(i + 1) + "_" + str(j + 1))
    for i in range(train_iter):
        header.append("q1_s"+ str(i + 1))
        header.append("q2_s"+ str(i + 1))
        header.append("q3_s"+ str(i + 1))
        header.append("width_s"+ str(i + 1))
        
    with file:
        write = csv.writer(file)
        write.writerow(header)
        for i in range(X.shape[1]): #X.shape[1]: total number of graphs
                                    #X.shape[0]: num_nodes^2
            G=nx.from_pandas_adjacency(pd.DataFrame(X[:,i].reshape(num_nodes, -1)))
            G=nx.Graph(G)
            col = np.random.randint(0,kappa,size=num_nodes)
            
            sample = []
            
            states, y = FCA(G, col, kappa, pred_iter)
            sample.append(y)

            baseline_width = width_compute(states[train_iter - 1], kappa)
            baseline = False
            if (baseline_width < floor(kappa / 2)):  # half circle concentration
                baseline = True
            sample.append(baseline)

            for j in range(train_iter):
                sample = sample + list(states[j])
            for j in range(train_iter):
                sample.append(s.quantiles(states[j], n=4)[0])
                sample.append(s.quantiles(states[j], n=4)[1])
                sample.append(s.quantiles(states[j], n=4)[2])
                sample.append(width_compute(states[j], kappa))
    
            write.writerow(sample)
        
#compute FCA width to check the half-circle concentration
#from L2PSync repo
def width_compute(coloring, kappa):
    differences = [np.max(coloring) - np.min(coloring)]
    for j in range(1,kappa+1):
        shifted = (np.array(coloring) + j) % kappa
        differences.append(np.max(shifted) - np.min(shifted))
    return np.min(differences)

def datagen_FCA_coladj(num_nodes, kappa, df_graph, df_dynamics, pred_iter, file_name):
    #df_graph: a list of vectorized adjacency matrix (for each graph, length = k^2 if num_nodes=k)
    #df_dynamics: the dataset generated by datagen_FCA_dynamics(); all columns need to be named
    num_nodes = 20
    kappa = 8
    df = pd.DataFrame()
    for i_graph in range(df_graph.shape[1]):
        adj_mx = df_graph[:,i_graph].reshape(num_nodes, -1)
        temp =[]
        for i_state in range(pred_iter):
            adj = adj_mx.copy()
            
            col = np.array(df_dynamics.loc[
                i_graph, 
                df_dynamics.columns[0+num_nodes*i_state]:df_dynamics.columns[19+num_nodes*i_state]
            ])

            for i in range(num_nodes):
                adj[i,i]=0
                for j in range(num_nodes):
                    if i<j:
                        adj[i,j]=min(abs(np.subtract(col[i], col[j], dtype=np.float32)),
                                     min(col[i], col[j])+kappa-max(col[i], col[j]))
                        adj[j,i]=min(abs(np.subtract(col[i], col[j], dtype=np.float32)),
                                     min(col[i], col[j])+kappa-max(col[i], col[j]))

            temp.append(adj.reshape(1,-1)[0])

        df = df.append(pd.Series(np.array(temp).reshape(1, -1)[0]), ignore_index=True)
    df.to_csv(file_name, index = False)

        
##############################GreenbergHasting##############################################################################
def GHM(G, s, k, iteration):
    ret = s
    s_next = np.zeros(G.number_of_nodes())
    
    for h in range(iteration):
        if h != 0:
            s = s_next # Update to the newest state
            ret = np.vstack((ret, s_next))
        s_next = np.zeros(G.number_of_nodes())
        for i in range(G.number_of_nodes()):
            if s[i] == 0:
                flag = True #if coloring of neighbor of 1 is not found
                for j in range(G.number_of_nodes()):
                    if s[j]==1 and list(G.nodes)[j] in list(G.adj[list(G.nodes)[i]]):
                        s_next[i] = 1
                        flag = False
                        break
                if flag:
                    s_next[i] = 0
            else:
                s_next[i] = (s[i]+1)%k
    
    label = False
    if np.sum(ret[-1]) == 0:
         label = True
        
    return ret, label

def datagen_GHM_dynamics(num_nodes, kappa, pred_iter, train_iter, file_name, X):
    """Generate GHM dynamics dataset

    Args:
        num_nodes: number of nodes of underlying graphs
        kappa: k-color GHM
        pred_iter: prediction iteration (label)
        train_iter: training iteration (baseline)
        file_name: output filename
        X: a list of vectorized adjacency matrix (for each graph, length = k^2 if num_nodes=k)

    Output:
        a csv file with label y, baseline_width
        Since GHM doesn't have width like FCA or Kuramoto, baseline and y are based on whether all colors actually synchronizes
        si_j: the coloring of node j at time i
        q1_si, q2_si, q3_si: the 1st, 2nd, 3rd quartile at time i
    """
    file = open(file_name, 'w+', newline='')
    header = ["y", "baseline_width"]
    for i in range(train_iter):
        for j in range(num_nodes):
            header.append("s" + str(i + 1) + "_" + str(j + 1))
    for i in range(train_iter):
        header.append("q1_s"+ str(i + 1))
        header.append("q2_s"+ str(i + 1))
        header.append("q3_s"+ str(i + 1))
        
    with file:
        write = csv.writer(file)
        write.writerow(header)
        for i in range(X.shape[1]): #X.shape[1]: total number of graphs
                                    #X.shape[0]: num_nodes^2
            G=nx.from_pandas_adjacency(pd.DataFrame(X[:,i].reshape(num_nodes, -1)))
            G=nx.Graph(G)
            col = np.random.randint(0,kappa,size=num_nodes)
            
            sample = []
            
            states, y = GHM(G, col, kappa, pred_iter)
            sample.append(y)

            baseline = False
            if np.sum(states[train_iter - 1]) == 0:  # half circle concentration
                baseline = True
            sample.append(baseline)

            for j in range(train_iter):
                sample = sample + list(states[j])
            for j in range(train_iter):
                sample.append(s.quantiles(states[j], n=4)[0])
                sample.append(s.quantiles(states[j], n=4)[1])
                sample.append(s.quantiles(states[j], n=4)[2])
    
            write.writerow(sample)
        
        
##############################KURAMOTO######################################################################################
def Kuramoto(G, K, s, iteration, step):
    ret = s
    s_next = np.zeros(G.number_of_nodes())
    for h in range(iteration-1):
        if h != 0:
            s = s_next # Update to the newest state
            ret = np.vstack((ret, s_next))
        for i in range(G.number_of_nodes()):
            neighbor_col = []
            for j in range(G.number_of_nodes()):
                if list(G.nodes)[j] in list(G.adj[list(G.nodes)[i]]):
                    neighbor_col.append(s[j])
            
            new_col = s[i]+ step * K * np.sum(np.sin(neighbor_col - s[i]))
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
    
def datagen_Kuramoto_dynamics(num_nodes, K, step, pred_iter, train_iter, skip, file_name, X):
    """Generate Kuramoto dynamics dataset

    Args:
        num_nodes: number of nodes of underlying graphs
        K: coupling strength
        step
        pred_iter: prediction iteration (label)
        train_iter: training iteration (baseline)
        skip: the number of iterations to skip (avoid large data files)
        file_name: output filename
        X: a list of vectorized adjacency matrix (for each graph, length = k^2 if num_nodes=k)

    Output:
        a csv file with label y, baseline_width
        si_j: the coloring of node j at time i
        q1_si, q2_si, q3_si: the 1st, 2nd, 3rd quartile at time i
        width_si: the width of time i
    """
    file = open(file_name, 'w+', newline='')
    header = ["y", "baseline_width"]
    for i in range(int(train_iter/skip)):
        for j in range(num_nodes):
            header.append("s" + str(i*skip + 1) + "_" + str(j + 1))
    for i in range(int(train_iter/skip)):
        header.append("q1_s"+ str(i*skip + 1))
        header.append("q2_s"+ str(i*skip + 1))
        header.append("q3_s"+ str(i*skip + 1))
        header.append("width_s"+ str(i*skip + 1))
        
    with file:
        write = csv.writer(file)
        write.writerow(header)
        for i in range(X.shape[1]): #X.shape[1]: total number of graphs
                                    #X.shape[0]: num_nodes^2
            G=nx.from_pandas_adjacency(pd.DataFrame(X[:,i].reshape(num_nodes, -1)))
            G=nx.Graph(G)
            col = np.random.uniform(-np.pi, np.pi, num_nodes)
            
            sample = []
            
            states, y = Kuramoto(G, K, col, pred_iter, step)
            sample.append(y)

            baseline_width = widthkura(states[train_iter - 1])
            baseline = False
            if (baseline_width < np.pi):  # half circle concentration
                baseline = True
            sample.append(baseline)

            for j in range(int(train_iter/skip)):
                sample = sample + list(states[j*skip])
            for j in range(int(train_iter/skip)):
                sample.append(s.quantiles(states[j*skip], n=4)[0])
                sample.append(s.quantiles(states[j*skip], n=4)[1])
                sample.append(s.quantiles(states[j*skip], n=4)[2])
                sample.append(widthkura(states[j*skip]))
    
            write.writerow(sample)

def datagen_Kuramoto_coladj(num_nodes, df_graph, df_dynamics, pred_iter, skip, file_name):
    #skip: the number of iterations to skip (avoid large data files)
    #df_graph: a list of vectorized adjacency matrix (for each graph, length = k^2 if num_nodes=k)
    #df_dynamics: the dataset generated by datagen_FCA_dynamics(); all columns need to be named
    num_nodes = 20
    kappa = 8
    df = pd.DataFrame()
    
    for i_graph in range(df_graph.shape[1]):
        adj_mx = df_graph[:,i_graph].reshape(num_nodes, -1)
        temp =[]
        for i_state in range(int(pred_iter/skip)):
            adj = adj_mx.copy()
            col = np.array(df_dynamics.loc[
                i_graph, 
                df_dynamics.columns[0+20*i_state]:df_dynamics.columns[19+20*i_state]
            ])

            for i in range(num_nodes):
                adj[i,i]=0
                for j in range(num_nodes):
                    if i<j:
                        new_col = np.subtract(max(col[i], col[j]), min(col[i], col[j]), dtype=np.float32)
                        if new_col > np.pi:
                            new_col = 2*np.pi - new_col
                        adj[i,j]=new_col
                        adj[j,i]=new_col

            temp.append(adj.reshape(1,-1)[0])

        df = df.append(pd.Series(np.array(temp).reshape(1, -1)[0]), ignore_index=True)
    df.to_csv(file_name, index = False)
        

#################################GRAPH#FEATURES#############################################################################
# X: a list of graph structure (for each graph, length = k^2 if num_nodes=k)
# num_edges, num_nodes, min_degree, max_degree, diameter
def datagen_graph_features(num_nodes, file_name, X):
    """Generate graph features dataset

    Args:
        num_nodes: number of nodes of underlying graphs
        file_name: output filename
        X: a list of vectorized adjacency matrix (for each graph, length = k^2 if num_nodes=k)

    Output:
        a csv file with features
        graph_level features: "num_edges", "num_nodes", "min_degree", "max_degree", "diameter", 
              "degree_assortativity_coef", "num_clique", "avg_clustering_coef", 
              "density"
        node_level features: "degree_centrality", "eigenvector_centrality", "betweenness_centrality", 
              "closeness_centrality", "clustering", "degree"
    """
    file = open(file_name, 'w+', newline='')

    header = ["num_edges", "num_nodes", "min_degree", "max_degree", "diameter", 
              "degree_assortativity_coef", "num_clique", "avg_clustering_coef", 
              "density"]
    for i in ["degree_centrality", "eigenvector_centrality", "betweenness_centrality", 
              "closeness_centrality", "clustering", "degree"]:
        for j in range(num_nodes):
            header.append(i + "_n" + str(j + 1))
    
    with file:
        write = csv.writer(file)

        write.writerow(header)
        for i in range(X.shape[1]): #X.shape[1]: total number of graphs
                                    #X.shape[0]: num_nodes^2
            G=nx.from_pandas_adjacency(pd.DataFrame(X[:,i].reshape(num_nodes, -1)))
            G=nx.Graph(G)

            num_edges = G.number_of_edges()
            min_degree = min(list(G.degree), key=lambda x: x[1])[1]
            max_degree = max(list(G.degree), key=lambda x: x[1])[1]
            diameter = nx.diameter(G)
            
            degree_assortativity_coef = nx.degree_assortativity_coefficient(G)
            num_clique = nx.graph_clique_number(G)
            avg_clustering_coef = nx.average_clustering(G)
            #small_world_coef = nx.omega(G)
            density = nx.density(G)

            sample = [num_edges, num_nodes, min_degree, max_degree, diameter,
                      degree_assortativity_coef, num_clique, avg_clustering_coef, 
                      density]
            
            node_feature = []
            node_feature.append(list(nx.degree_centrality(G).values()))
            node_feature.append(list(nx.eigenvector_centrality(G, tol=1.0e-3).values()))
            node_feature.append(list(nx.betweenness_centrality(G).values()))
            node_feature.append(list(nx.closeness_centrality(G).values()))
            node_feature.append(list(nx.clustering(G).values()))
            node_feature.append(list(dict(G.degree()).values()))
            sample = sample + list(np.array(node_feature).reshape(1, -1)[0])

            write.writerow(sample)


def datagen_n2v(num_nodes, file_name, dimensions, X):
    df = pd.DataFrame()
    
    header = []
    for i in range(num_nodes):
        for j in range(dimensions):
            header.append("n"+str(i)+"d"+str(j))
            
    for i in range(X.shape[1]):
        G=nx.from_pandas_adjacency(pd.DataFrame(X[:,i].reshape(num_nodes, -1)))
        G=nx.Graph(G)
        node2vec_model = Node2Vec(dimensions = dimensions)
        node2vec_model.fit(G)
        df = df.append(pd.Series(node2vec_model.get_embedding().reshape(1, -1)[0]), ignore_index=True)
    df.columns = header
    df.to_csv(file_name, index = False)
    
    
def datagen_g2v(num_nodes, file_name, dimensions, X):
    df = pd.DataFrame()
    for i in range(X.shape[1]):
        G=nx.from_pandas_adjacency(pd.DataFrame(X[:,i].reshape(num_nodes, -1)))
        G=nx.Graph(G)
        graph2vec_model = Graph2Vec(dimensions=dimensions, min_count = 1)
        graph2vec_model.fit([G])
        df = df.append(pd.Series(graph2vec_model.get_embedding()[0]), ignore_index=True)
    df.to_csv(file_name, index = False)

#################################GRAPH#FEATURES#############################################################################