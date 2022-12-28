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
from SDL import *

# =======================================================
# Construct Argument Parser
# =======================================================
parser = argparse.ArgumentParser()    
parser.add_argument("--method", default="supervised", type=str,
                        help="Either 'supervised' or 'unsupervised'")
args = parser.parse_args()

assert args.method in ['supervised', 'unsupervised']

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


# For supervised data-informed approach -> Just apply SDL directly
if args.method == 'supervised':


# For unsupervised data-informed approach -> Use NMF + Logistic Regression
if args.method == 'unsupervised':




