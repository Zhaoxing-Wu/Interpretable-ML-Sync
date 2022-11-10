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
dynamic = "ghm"
samples = 10000
k = 15
name = "motifDynamics/SAMPLES-"+str(samples)+"_NTWK-"+ntwk+"_K-"+str(k)+'_DYNAMIC-'+str(dynamic)+'.pkl'
# pickle load the object
motif = pickle.loads(s3_bucket.Object(name).get()['Body'].read())

import pdb; pdb.set_trace()
