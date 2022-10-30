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
objects = s3_client.list_objects_v2(Bucket='interpretable-sync')
allkeys = [obj['Key'] for obj in objects['Contents']]

# read in file from AWS
ntwk = "Caltech36"
samples = 10000
k = 15
filename = "motifSampling/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(k)+"_PATCHES.pkl"
# pickle load the object
xEmbDes = pickle.loads(s3_bucket.Object(filename).get()['Body'].read())
adjmats = xEmbDes['X']

# set of dynamics
dynamics = {"fca": [datagen.FCA], "ghm": [datagen.GHM], "kura":[datagen.Kuramoto]}
for dynamic in ["fca","ghm","kura"]:

    # iterate through each motif
    data_to_store = []
    for mat in tqdm.tqdm(adjmats.transpose()):

        # parameters
        kappa = 5
        iterations = 20
        stepKura = 0.0001
        kuraK = 1.34

        # create adjacency matrix
        submat = mat.reshape(k,k)
        # create graph
        G = nx.from_numpy_matrix(submat)
        # initial colors
        if dynamics=="kura":
            cols = np.random.rand(1, k)*2*np.pi-np.pi
        else:
            cols = np.random.randint(0, kappa, k)

        # simulation
        if dynamic=="kura":
            ret, label = dynamics[dynamic][0](G, kuraK, cols, iterations, stepKura)
        else:
            ret, label = dynamics[dynamic][0](G, cols, kappa, iterations)
        colorsNlabels = [ret, G, label]

        # append
        data_to_store.append(colorsNlabels) 

    # file key --- file writing
    name = "motifDynamics/SAMPLES-"+str(samples)+"_NTWK-"+ntwk+"_K-"+str(k)+'_DYNAMIC-'+str(dynamic)+'.pkl'
    if name not in allkeys:
        # =======================================================
        # Object -> binary stream -> bucket
        # =======================================================
        # Use dumps() to make it serialized
        binary_stream = pickle.dumps(data_to_store)
        # dump in bucket
        print("'"+name+"' was added to the bucket.")
        s3_bucket.put_object(Body=binary_stream, Key=name)
        # update allkeys
        objects = s3_client.list_objects_v2(Bucket='interpretable-sync')
        allkeys = [obj['Key'] for obj in objects['Contents']]
    else:
        print(name+" already exists.")
