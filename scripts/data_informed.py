# Importing basic libraries required:
import os
import tqdm
import pickle
import argparse
import numpy as np
import pandas as pd

# Importing scikit-learn libraries:
from sklearn.linear_model import LogisticRegression

# Importing network-based libraries:
from NNetwork import NNetwork as nn
import networkx as nx

# Importing AWS Cloud S3 Bucket Library:
import boto3

# Importing custon libraries:
from nmf import *
from SDL import SDL_BCD

# =======================================================
# Construct Argument Parser
# =======================================================
parser = argparse.ArgumentParser(description='Evaluates accuracies for classification of dynamics',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
parser.add_argument("--method", default="supervised", type=str,
                        choices=['supervised', 'unsupervised'], help="Whether to use a supervised or unsupervised approach")
parser.add_argument("--model", default="kura", type=str,
                        choices=['kura', 'fca', 'ghm'], help="Which model of coupled oscillator to use")
parser.add_argument("--num_dicts", default=8, type=int,
                        help="Number of dictionary atoms to learn")
parser.add_argument("--logs_dir", default="../results", type=str,
                        help="Folder to store results from runs")
parser.add_argument("--DEBUG", default=False, type=bool,
                        help="Whether to stay on debug mode")
args = parser.parse_args()

ref_df = pd.read_csv('../Drivers/datagen/ca_params.csv') # DataFrame to use for training SDL later
ref_df = ref_df.set_index("ntwk")

# =======================================================
# Connect to S3 resource
# =======================================================
# Connect to S3 and create resource object:
s3_resource = boto3.resource(
            service_name='s3',
            region_name='us-west-1',
            aws_access_key_id='AKIAWNJSAXHUWYXA4YJF',
            aws_secret_access_key='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'
            )
s3_client = boto3.client('s3',
            aws_access_key_id='AKIAWNJSAXHUWYXA4YJF',
            aws_secret_access_key='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')

# Specify bucket object:
s3_bucket = s3_resource.Bucket('interpretable-sync')  # type: ignore
objects = s3_client.list_objects_v2(Bucket='interpretable-sync')
allkeys = [obj['Key'] for obj in objects['Contents']]

# For supervised data-informed approach -> Use SDL directly
if args.method == 'supervised':
    for k in range(10, 31, 5):
        i = 0
        for ntwk in ['Caltech36', 'nws-20000-1000-05', 'UCLA26']:
            dynamics = f"motifDynamics_new/SAMPLES-10000_NTWK-{ntwk}_K-{k}_DYNAMIC-{args.model}_PARAMS-csv.pkl"
            col_adj = f"motifDynamics_new/SAMPLES-10000_NTWK-{ntwk}_K-{k}_COLADJ-{args.model}_PARAMS-csv.pkl"

            if args.DEBUG:
                print(dynamics)
                print(col_adj)
            
            X = pickle.loads(s3_bucket.Object(dynamics).get()['Body'].read()) # Dynamics-label Dataset
            CCAT = pickle.loads(s3_bucket.Object(col_adj).get()['Body'].read()) # Color-Coded Adjacency Tensor

            df_dyn = pd.DataFrame(X)
            df_ccat = pd.DataFrame(CCAT)

            if args.DEBUG:
                print(f"For k={k}, ntwk={ntwk}: Dynamics- {df_dyn.shape}")
                print(f"For k={k}, ntwk={ntwk}: CCAT- {df_ccat.shape}")

            y = df_dyn.y
            base = df_dyn.baseline_width
            s_ind = f'{args.model}_train_iter'
            Y = df_dyn["y"].values
            # df_dyn = df_dyn.loc[:, 's1_1':f"s{ref_df.loc['ntwk', s_ind][i]}_{k}"] # type: ignore
            X = df_ccat.T.values

            xi = 1
            iter_avg = 1
            beta = 0.5
            iteration = 100
            r = 4
            SDL_BCD_class_new = SDL_BCD(X=[X, Y],  # data, label
                        X_test=[X_test.T, y_test.to_numpy().reshape(-1,1).T],
                        #X_auxiliary = None,
                        n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                        # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                        # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                        # ini_code = H_true,
                        xi=xi,  # weight on label reconstruction error
                        L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                        L2_reg = [0,0,0], # L2 regularizer for code H, dictionary W[0], reg param W[1]
                        nonnegativity=[True,True,False], # nonnegativity constraints on code H, dictionary W[0], reg params W[1]
                        full_dim=False)



# For unsupervised data-informed approach -> Use NMF + Logistic Regression
if args.method == 'unsupervised':
    for k in range(10, 31, 5):
        for ntwk in ['Caltech36', 'nws-20000-1000-05', 'UCLA26']:
            col_adj = f"motifDynamics_new/SAMPLES-10000_NTWK-{ntwk}_K-{k}_COLADJ-{args.model}_PARAMS-csv.pkl"
            X = pickle.loads(s3_bucket.Object(col_adj).get()['Body'].read())
            df = pd.DataFrame(X)
            print(df.shape)