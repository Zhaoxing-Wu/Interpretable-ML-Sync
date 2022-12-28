# Imports
import numpy as np
from NNetwork import NNetwork as nn
import networkx as nx
import tqdm
import os

# Cloud 
import boto3
import pickle
import pdb

# =======================================================
# Connect to S3 resource
# =======================================================

# Connect to S3 and create resource object
s3_resource = boto3.resource(
            service_name='s3',
            region_name='us-west-1',
            aws_access_key_id='AKIAWNJSAXHUWYXA4YJF',
            aws_secret_access_key='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'
            )
s3_client = boto3.client('s3', 
            aws_access_key_id='AKIAWNJSAXHUWYXA4YJF',
            aws_secret_access_key='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')

# Specify bucket object
s3_bucket = s3_resource.Bucket('interpretable-sync') # type: ignore
objects = s3_client.list_objects_v2(Bucket='interpretable-sync')
allkeys = [obj['Key'] for obj in objects['Contents']]

# Print bucket contents
if 0:
    for s3_bucket_object in s3_bucket.objects.all():
        print(s3_bucket_object.key)

# List of networks
names = []
with os.scandir('../Networks_all_NDL') as entries:
    for entry in entries:
        names.append(entry.name.split(".")[0])

print("Number of networks: ", len(names))
print("Names of networks:", names)
# pdb.set_trace()

# For each network
#for ntwk in tqdm.tqdm(names[15:]):
for ntwk in tqdm.tqdm(names[15:]):

    # Load Network
    sampling_alg = 'pivot'
    samples = 10000
    ntwk_nonumber = ''.join([i for i in ntwk if not i.isdigit()])
    path = "../Networks_all_NDL/" + ntwk + '.txt'
    
    # Create graph
    G = nn.NNetwork()
    G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
    if 0:
        print('num nodes in G', len(G.nodes()))
        print('num edges in G', len(G.get_edges()))

    # Sample k paths
    for k in [5, 10, 15, 20, 25, 30]:
        # Print progress
        if 1:
            print("Working on "+ntwk+" of k="+str(k))

        # File key
        name = "motifSampling/SAMPLES-"+str(samples)+"_NTWK-"+ntwk+"_K-"+str(k)+'_PATCHES.pkl'
        if name not in allkeys:
                # Generate patches
                xembds = G.get_patches(k=k, sample_size=samples, skip_folded_hom=True)
                # Saving data to store
                data_to_store = {"X"    : xembds[0],
                                "embs" : xembds[1],
                                "Description" : "Motif sampling data on "
                                +str(samples)+" sample of network '"+ntwk
                                +"' of Hamiltonian path of length "+str(k)}
                
                # =======================================================
                # Object -> binary stream -> bucket
                # =======================================================
                
                # Use dumps() to make it serialized
                binary_stream = pickle.dumps(data_to_store)
                # Dump in bucket
                print("'"+name+"' was added to the bucket.")
                s3_bucket.put_object(Body=binary_stream, Key=name)
                # Update allkeys
                objects = s3_client.list_objects_v2(Bucket='interpretable-sync')
                allkeys = [obj['Key'] for obj in objects['Contents']]
   
        
        else:
            print(name+" already exists.")
            continue
