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
args = parser.parse_args()

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

# Read files containing dynamics from S3 bucket:
# binary_stream = s3_resource.Object(bucket_name='interpretable-sync', 
#         key='dummy_folder-100922/dummy_copy-100922.pkl').get()
# data = pickle.loads(binary_stream['Body'].read())

# For supervised data-informed approach -> Use SDL directly
if args.method == 'supervised':
    for k in range(10, 31, 5):
        for ntwk in ['Caltech36', 'nws-20000-1000-05', 'UCLA26']:
            col_adj = f"motifDynamics/SAMPLES-10000_NTWK-{ntwk}_K-{k}_COLADJ-{args.model}_PARAMS-csv.pkl"
            X = pickle.loads(s3_bucket.Object(col_adj).get()['Body'].read())
            df = pd.DataFrame(X)
            print(df.shape)

# For unsupervised data-informed approach -> Use NMF + Logistic Regression
if args.method == 'unsupervised':
    for k in range(10, 31, 5):
        for ntwk in ['Caltech36', 'nws-20000-1000-05', 'UCLA26']:
            col_adj = f"motifDynamics/SAMPLES-10000_NTWK-{ntwk}_K-{k}_COLADJ-{args.model}_PARAMS-csv.pkl"
            X = pickle.loads(s3_bucket.Object(col_adj).get()['Body'].read())
            df = pd.DataFrame(X)
            print(df.shape)