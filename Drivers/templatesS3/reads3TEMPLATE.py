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

# =======================================================
# Check bucket contents
# =======================================================
# specify bucket object
s3_bucket = s3_resource.Bucket(name='interpretable-sync')
if 0:
    # print bucket contents
    for s3_bucket_object in s3_bucket.objects.all():
        print(s3_bucket_object.key)

# =======================================================
# bucket -> binary stream -> data object
# =======================================================
binary_stream = s3_resource.Object(bucket_name='interpretable-sync', 
        key='dummy_folder-100922/dummy_copy-100922.pkl').get()
data = pickle.loads(binary_stream['Body'].read())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# *** rest of the script ***
print(data)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
