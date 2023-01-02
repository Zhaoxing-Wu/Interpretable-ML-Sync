import pickle, csv, random
import numpy as np
import pandas as pd
import statistics as s
from itertools import product
from math import floor
from tqdm import trange
from scipy.stats import bernoulli
import networkx as nx
from NNetwork import NNetwork as nn
from NNetwork.NNetwork import NNetwork

from sklearn import svm, metrics, model_selection
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.decomposition import PCA ### Use truncated SVD / online PCA later for better computational efficiency
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from karateclub import Graph2Vec, Node2Vec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from nmf import *
warnings.filterwarnings("ignore")

def select_repr_ex(df_features):
    ind = []
    ind.append(sorted(range(len(df_features.num_edges)), key=lambda i: df_features.num_edges[i], reverse=True)[0])
    temp = sorted(range(len(df_features.num_clique)), key=lambda i: df_features.num_clique[i], reverse=True)
    for i in temp:
        if i not in ind:
            ind.append(i)
            break
    temp = sorted(range(len(df_features.diameter)), key=lambda i: df_features.num_clique[i], reverse=False)
    for i in temp:
        if i not in ind:
            ind.append(i)
            break
    ind.append(sorted(range(len(df_features.num_edges)), key=lambda i: df_features.num_edges[i], reverse=False)[0])
    temp = sorted(range(len(df_features.num_clique)), key=lambda i: df_features.num_clique[i], reverse=False)
    for i in temp:
        if i not in ind:
            ind.append(i)
            break
    temp = sorted(range(len(df_features.diameter)), key=lambda i: df_features.num_clique[i], reverse=True)
    for i in temp:
        if i not in ind:
            ind.append(i)
            break
    
    ind_title = ["Largest #edges", "Largest #cliques", "Smallest diameter",
             "Smallest #edges", "Smallest # cliques","Largest diameter"]
    return ind, ind_title

#color edge weight
#plot what iteration
def plot_repr_ex(df_features, df_graph, num_nodes, results_dict_new, X_train, y_train,
                iter_arr):
    ind, ind_title = select_repr_ex(df_features)
    ncol = len(ind)
    nrow = len(iter_arr)+1
    fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(ncol*4, nrow*4))
    
    W = results_dict_new["loading"][0]

    for i in range(ncol):
        ind_data = list(y_train.index).index(ind[i])
        lg = LogitRegression(learning_rate = 0.01, iterations = 1000)
        #X_filtered = W^TX
        x = np.array([np.matmul(W.T, 
                                X_train[list(y_train.index).index(ind[i]),:].T).T])
        lg.fit(x, np.array(y_train[ind[i]]))
        axs[i//ncol, i%ncol].title.set_text(ind_title[i]+" ("+str(np.array(y_train[ind[i]])) + ")")
        axs[i//ncol, i%ncol].imshow(np.array([lg.W[coef_idx]]))

        df_adj = df_graph[:, ind[i]].reshape(20, 20)
        G = nx.Graph()
        G = nx.from_pandas_adjacency(pd.DataFrame(df_adj))
        for j in range(len(iter_arr)):
            x_t = np.array([np.matmul(W[(0):(400+iter_arr[j]*400),:].T,
                                      X_train[ind_data, (0):(400+iter_arr[j]*400)].T).T])

            axs[i//ncol+1+j, i%ncol].title.set_text(model.predict_prob(x_t))

            G1 = nx.Graph()
            for a in range(num_nodes):
                for b in range(num_nodes):
                    u = list(G.nodes())[a]
                    v = list(G.nodes())[b]
                    if G.has_edge(u,v) and u!=v:
                        if abs(X_train[ind_data, 
                              (0+iter_arr[j]*400):(400+iter_arr[j]*400)].reshape(-1, num_nodes)[u, v])==0: #< 0.03: #all synchronizing edges
                            G1.add_edge(u,v, color='r')
                        else:
                            G1.add_edge(u,v, color='b')

            edges = G1.edges()
            colors = [G1[u][v]['color'] for u,v in edges]
            nx.draw(G1, edge_color=colors, node_size= 50, ax=axs[i//ncol+1+j, i%ncol], 
                    pos = nx.spring_layout(G1, seed=123))



###################################################################################################
#r: number of dictionary elements
#diameter
#base: baseline_width
#X: coloring adj
def dict_handcraft(diameter, base, X, r):
    #select the indexes for samples with corresponding features (dense, sparse, concentrate)
    sample_size = 100 #number of samples used for learning dictionary
    ind_dense = sorted(range(len(diameter)), key=lambda i: diameter[i], reverse=True)[:sample_size]
    ind_sparse = sorted(range(len(diameter)), key=lambda i: diameter[i], reverse=False)[:sample_size]
    ind_con = X.loc[base==True,].index.tolist()
    if len(ind_con)>sample_size:
        ind_con = ind_con[:sample_size]
    
    W_dense, H_dense = ALS(X=X.loc[ind_dense,].T.values, 
                           n_components=r, n_iter=100, a0 = 0, a1 = 0, a12 = 0, H_nonnegativity=True, 
                           W_nonnegativity=True, compute_recons_error=True, subsample_ratio=1)
    W_sparse, H_sparse = ALS(X=X.loc[ind_sparse,].T.values, 
                             n_components=r, n_iter=100, a0 = 0, a1 = 0, a12 = 0, 
                             H_nonnegativity=True, W_nonnegativity=True, 
                             compute_recons_error=True, subsample_ratio=1)
    W_con, H_con = ALS(X=X.loc[ind_con,].T.values, n_components=r, 
                       n_iter=100, a0 = 0, a1 = 0, a12 = 0, H_nonnegativity=True, 
                       W_nonnegativity=True, compute_recons_error=True, subsample_ratio=1)

    return np.concatenate([W_dense.T, W_sparse.T, W_con.T])