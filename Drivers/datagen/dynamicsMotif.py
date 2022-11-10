# Imports
import numpy as np
import pandas as pd
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
df = pd.read_csv('ca_param.csv')
# Caltech
caltech = {
        'fca': df[df['ntwk']=="Caltech36"][["fca_pred_iter","fca_train_iter","fca_kappa"]],
        'kura': df[df['ntwk']=="Caltech36"][["kura_pred_iter","kura_train_iter","kura_k",
            "kura_step","kura_skip"]],
        'ghm': df[df['ntwk']=="Caltech36"][["ghm_pred_iter","ghm_train_iter","ghm_kappa"]]
        }

# UCLA
ucla = {
        'fca': df[df['ntwk']=="UCLA26"][["fca_pred_iter","fca_train_iter","fca_kappa"]],
        'kura': df[df['ntwk']=="UCLA26"][["kura_pred_iter","kura_train_iter","kura_k",
            "kura_step","kura_skip"]],
        'ghm': df[df['ntwk']=="UCLA26"][["ghm_pred_iter","ghm_train_iter","ghm_kappa"]]
        }

# nws
nws = {
        'fca': df[df['ntwk']=="nws"][["fca_pred_iter","fca_train_iter","fca_kappa"]],
        'kura': df[df['ntwk']=="nws"][["kura_pred_iter","kura_train_iter","kura_k",
            "kura_step","kura_skip"]],
        'ghm': df[df['ntwk']=="nws"][["ghm_pred_iter","ghm_train_iter","ghm_kappa"]]
        }

# parameters
params = {
        'Caltech36': caltech, 
        'UCLA26': ucla,
        'nws-20000-1000-05': nws
        }

# names of networks
names = ['nws-20000-1000-05', 'UCLA26', 'Caltech36']

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

for ntwk in names:
    # read in file from AWS
    samples = 10000
    for i,k in enumerate([10, 15, 20, 25, 30]):
        filename = "motifSampling/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(k)+"_PATCHES.pkl"
        # pickle load the object
        xEmbDes = pickle.loads(s3_bucket.Object(filename).get()['Body'].read())
        
        # "Bella threw away other returns"
        if ntwk == "nws-20000-1000-05":
            adjmats = xEmbDes
        else:
            adjmats = xEmbDes['X']

        # set of dynamics
        dynamics = {"fca": datagen.FCA, "ghm": datagen.GHM, "kura": datagen.Kuramoto}
        for dynamic in ["kura","fca","ghm"]:

            # simulation parameters
            pntwk = params[ntwk][dynamic].copy().reset_index()

            # file key --- file writing
            name = "motifDynamics/SAMPLES-"+str(samples)+"_NTWK-"+ntwk+"_K-"+str(k)+'_DYNAMIC-'+str(dynamic)+'_PARAMS-csv.pkl'
            if name not in allkeys:
                # ================================================================================================
                # ================================================================================================
                # ================================================================================================
                # iterate through each motif
                data_to_store = []
                for mat in tqdm.tqdm(adjmats.transpose()):

                    # create adjacency matrix
                    submat = mat.reshape(k,k)
                    # create graph
                    G = nx.from_numpy_matrix(submat)
                    
                    # simulation
                    if dynamic=="kura":
                        cols = np.random.rand(k)*2*np.pi-np.pi
                        ret, label = dynamics[dynamic](G, pntwk['kura_k'][i], cols, pntwk['kura_train_iter'][i], pntwk['kura_step'][i])

                        # skipping subsample

                        # test label

                        # baseline width
                    elif dynamic=="fca":
                        cols = np.random.randint(0, pntwk['fca_kappa'][i], k )
                        ret, label = dynamics[dynamic](G, cols, pntwk['fca_kappa'][i], pntwk['fca_train_iter'][i])
                        # test label

                        # baseline width
                    else:
                        cols = np.random.randint(0, pntwk['ghm_kappa'][i], k)
                        ret, label = dynamics[dynamic](G, cols, pntwk['ghm_kappa'][i], pntwk['ghm_train_iter'][i])
                        # test label

                        # baseline width
                    colorsNlabels = [ret, G, label]

                # append
                data_to_store.append(colorsNlabels) 
                # ================================================================================================
                # ================================================================================================
                # ================================================================================================

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

            # free memory
            del pntwk            
