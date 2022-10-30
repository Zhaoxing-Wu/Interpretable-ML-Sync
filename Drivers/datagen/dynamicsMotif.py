# Imports
import numpy as np
from NNetwork import NNetwork as nn
import networkx as nx
import datagen
import tqdm
import os
# Cloud 
import boto3
import pickle
# =======================================================
# Connect to S3 resource
# =======================================================
# connect to S3 and create resource object
s3_resource = boto3.resource(
            service_name='s3',
            region_name='us-west-1',
            aws_access_key_id='AKIAWNJSAXHUWYXA4YJF',
            aws_secret_access_key='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'
            )
s3_client = boto3.client('s3', 
            aws_access_key_id='AKIAWNJSAXHUWYXA4YJF',
            aws_secret_access_key='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')

# specify bucket object
s3_bucket = s3_resource.Bucket('interpretable-sync')

# read in file
ntwk = "Caltech36"
k = 15
filename = "motifSampling/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(k)+"_PATCHES.pkl"

# pickle load the object
xEmbDes = pickle.loads(s3_bucket.Object(filename).get()['Body'].read())
adjmats = xEmbDes['X']
for mat in tqdm.tqdm(adjmats.transpose()):

    # coloring
    kappa = 5

    # create adjacency matrix
    submat = mat.reshape(k,k)
    # create graph
    G = nx.from_numpy_matrix(submat)
    # initial colors
    cols = np.random.randint(0, kappa, k)

    # simulation
    colorsNlabels = datagen.FCA(G, kappa, 1000)

    # append


