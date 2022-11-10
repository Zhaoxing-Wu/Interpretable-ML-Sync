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

df = pd.read_csv('ca_param.csv')
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
    samples = 10000
    for i,k in enumerate([25, 30]):
        #read X
        filename = "motifSampling/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(k)+"_PATCHES.pkl"
        xEmbDes = pickle.loads(s3_bucket.Object(filename).get()['Body'].read())
        if ntwk == "nws-20000-1000-05":
            X = xEmbDes
        else:
            X = xEmbDes['X']
            
        num_nodes = k

        # set of dynamics
        for dynamic in ["kura","fca","ghm"]:
            
            #output name
            name_dynamics = "motifDynamics/SAMPLES-"+str(samples)+"_NTWK-"+ntwk+"_K-"+str(k)+'_DYNAMIC-'+str(dynamic)+'_PARAMS-csv.pkl'
            name_coladj = "motifDynamics/SAMPLES-"+str(samples)+"_NTWK-"+ntwk+"_K-"+str(k)+'_COLADJ-'+str(dynamic)+'_PARAMS-csv.pkl'
            
            if name_dynamics not in allkeys:
                print("pass")
                if dynamic=="kura":
                    df_dynamics = datagen.datagen_Kuramoto_dynamics(num_nodes, 
                                              df[df['ntwk']==ntwk][df['k']==k][["kura_k"]].values[0][0], #k
                                              df[df['ntwk']==ntwk][df['k']==k][["kura_step"]].values[0][0], #step 
                                              df[df['ntwk']==ntwk][df['k']==k][["kura_pred_iter"]].values[0][0], #pred_iter 
                                              df[df['ntwk']==ntwk][df['k']==k][["kura_train_iter"]].values[0][0], #train_iter 
                                              df[df['ntwk']==ntwk][df['k']==k][["kura_skip"]].values[0][0], #skip
                                              X)
                    df_coladj = datagen.datagen_Kuramoto_coladj(num_nodes, 
                                                        X, 
                                                        df_dynamics)
                    
                    
                elif dynamic=="fca":
                    df_dynamics = datagen.datagen_FCA_dynamics(num_nodes, 
                                         df[df['ntwk']==ntwk][df['k']==k][["fca_kappa"]].values[0][0],
                                         df[df['ntwk']==ntwk][df['k']==k][["fca_pred_iter"]].values[0][0],
                                         df[df['ntwk']==ntwk][df['k']==k][["fca_train_iter"]].values[0][0],
                                         X)
                    df_coladj = datagen.datagen_coladj(num_nodes, 
                                   df[df['ntwk']==ntwk][df['k']==k][["fca_kappa"]].values[0][0], 
                                   X, 
                                   df_dynamics)
                        
                else:#ghm
                    df_dynamics = datagen.datagen_GHM_dynamics(num_nodes, 
                                         df[df['ntwk']==ntwk][df['k']==k][["ghm_kappa"]].values[0][0],
                                         df[df['ntwk']==ntwk][df['k']==k][["ghm_pred_iter"]].values[0][0],
                                         df[df['ntwk']==ntwk][df['k']==k][["ghm_train_iter"]].values[0][0],
                                         X)
                    df_coladj = datagen.datagen_coladj(num_nodes, 
                                   df[df['ntwk']==ntwk][df['k']==k][["ghm_kappa"]].values[0][0], 
                                   X, 
                                   df_dynamics)
                
                
                binary_stream = pickle.dumps(df_dynamics)
                print("'"+name_dynamics+"' was added to the bucket.")
                s3_bucket.put_object(Body=binary_stream, Key=name_dynamics)
                
                binary_stream = pickle.dumps(df_coladj)
                print("'"+name_coladj+"' was added to the bucket.")
                s3_bucket.put_object(Body=binary_stream, Key=name_coladj)
                # update allkeys
                objects = s3_client.list_objects_v2(Bucket='interpretable-sync')
                allkeys = [obj['Key'] for obj in objects['Contents']]
                
            else:
                print(name_dynamics+" already exists.")
            
            #del pntwk # free memory