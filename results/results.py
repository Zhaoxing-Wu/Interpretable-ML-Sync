import boto3
import pickle
import csv

s3_resource = boto3.resource(
            service_name='s3',
            region_name='us-west-1',
            aws_access_key_id='AKIAWNJSAXHUWYXA4YJF',
            aws_secret_access_key='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'
            )

s3_bucket = s3_resource.Bucket(name='interpretable-sync')
                
for ca in ["kura", "fca"]:
    for ntwk in ['nws-20000-1000-05', 'Caltech36']:
        for run in range(5):
            with open(f'{ca}_{ntwk}_{run+1}.csv', 'w') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(['# Dictionaries', 'Acc10', 'Acc15', 'Acc20', 'Acc25', 'Acc30'])
                for r in [2, 4, 8, 16]:
                    table_row = []
                    table_row.append(f"r{r}")
                    for num_nodes in [10, 15, 20, 25, 30]:
                        dictionary = pickle.loads(s3_bucket.Object(f"output/sdl_runs/{ca}/r{r}/SAMPLES-10000_NTWK-{ntwk}_K-{num_nodes}_DYNAMIC-{ca}_sdl-r{r}_{run+1}.pkl")
                                                    .get()['Body']
                                                    .read())
                        table_row.append(dictionary["Accuracy"])
                    writer.writerow(table_row)