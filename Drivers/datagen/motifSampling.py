# Imports
import numpy as np
from NNetwork import NNetwork as nn
import networkx as nx
#import utils.NNetwork as nn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics, model_selection
from tqdm import trange
from sklearn.cluster import KMeans
import matplotlib.gridspec as gridspec
# Use truncated SVD / online PCA later for better computational efficiency
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

networklist = ['a', 'b']
for network in networklist:

    # load network
    sampling_alg = 'pivot'
    ntwk = 'Caltech36'  # COVID_PPI, Wisconsin87, UCLA26
    ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
    save_folder = 'Subgraph_sampling/test1/'
    k = 20

    # create graph
    path = "Data/Networks_all_NDL/" + str(ntwk) + '.txt'
    G = nn.NNetwork()
    G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
    print('num nodes in G', len(G.nodes()))
    print('num edges in G', len(G.get_edges()))

    # generate patches
    X, embs = G.get_patches(k=k, sample_size=10000, skip_folded_hom=True)

    # upload patches to S3 bucket (on the fly)
