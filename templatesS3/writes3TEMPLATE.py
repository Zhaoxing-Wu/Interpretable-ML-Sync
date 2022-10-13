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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# *** data generation ***
# Generate data object 
import pandas as pd
# Make dataframes
data_to_store = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# =======================================================
# Object -> binary stream -> bucket
# =======================================================
# specify bucket object
s3_bucket = s3_resource.Bucket('interpretable-sync')
# Use dumps() to make it serialized
binary_stream = pickle.dumps(data_to_store)
# folder placement
s3_bucket.put_object(Body=binary_stream, Key='dummy_folder-100922/dummy_copy-100922.pkl')
# root folder placement
s3_bucket.put_object(Body=binary_stream, Key='dummy-100922.pkl')

# =======================================================
# Check bucket contents
# =======================================================
if 1:
    # print bucket contents
    for s3_bucket_object in s3_bucket.objects.all():
        print(s3_bucket_object.key)
