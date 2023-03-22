from helper import *
import boto3
import pkg_resources
print(pkg_resources.get_distribution("pandas").version)
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

"""
for ntwk in ['nws-20000-1000-05']:
    for num_nodes in [10, 15, 20, 25, 30]:
        #read X
        ntwk_filename = "motifSampling/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+"_PATCHES.pkl"
        feature_filename = "motifSampling/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+"_graph_features.csv"
        xEmbDes = pickle.loads(s3_bucket.Object(ntwk_filename).get()['Body'].read())
        df_feature = pickle.loads(s3_bucket.Object(feature_filename).get()['Body'].read())
        if ntwk == "nws-20000-1000-05":
            X = xEmbDes
        else:
            X = xEmbDes['X']
"""

#filename = "SAMPLES-10000_NTWK-nws-20000-1000-05_K-30_DYNAMIC-fca_sdl-r8.pkl"
#filename = "SAMPLES-10000_NTWK-Caltech36_K-20_DYNAMIC-fca_sdl-r8.pkl"
filename = "SAMPLES-10000_NTWK-nws-20000-1000-05_K-20_DYNAMIC-kura_sdl-r8.pkl"
results_dict_new = pd.read_pickle(filename)

ncol = 8
nrow = 4
num_nodes = 20
fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(ncol*4, nrow*4))
sorted_indices = np.argsort(results_dict_new["loading"][1][0][1:])[::-1]
for i in range(ncol):
    print(i)
    ind = sorted_indices[i]
    df_adj = pd.DataFrame(results_dict_new["loading"][0].T[ind][0:num_nodes**2].reshape(-1, num_nodes))
    G = nx.from_pandas_adjacency(df_adj)
    print(results_dict_new["loading"][0].T[ind].shape)
    for j in range(nrow):
        col_adj = results_dict_new["loading"][0].T[ind][0+j*6*num_nodes**2:num_nodes**2+j*6*num_nodes**2].reshape(-1, num_nodes)
        
        G1 = nx.Graph()
        for a in range(num_nodes):
            for b in range(num_nodes):
                u = list(G.nodes())[a]
                v = list(G.nodes())[b]
                if G.has_edge(u,v) and u!=v:
                    if col_adj[u, v] <= 0.0002: #all synchronizing edges
                        G1.add_edge(u,v, color='r')
                    else:
                        G1.add_edge(u,v, color='b')
        edges = G1.edges()
        colors = [G1[u][v]['color'] for u,v in edges]
        weights = [220 * G[u][v]['weight'] for u, v in edges]
        nx.draw(G1, ax = axs[j, i], edge_color=colors, pos = nx.spring_layout(G1, seed=123),width=weights,
                node_size = 100)
        #sns.heatmap(results_dict_new["loading"][0].T[ind][0+j*10*400:400+j*10*400].reshape(-1, 20),
                    #ax = axs[j, i])
        if j == 0:
            axs[j, i].title.set_text(str(round(results_dict_new["loading"][1][0][1:][sorted_indices[i]], 3)))
        #else:
            #axs[j, i].title.set_text("time"+str(j*10))

        
    #rect = plt.Rectangle((0.12+i*0.1, 0.12), 0.09, 0.78, fill=False, color="k", lw=1,
                         #zorder=1000, transform=fig.transFigure, figure=fig)
    #fig.patches.extend([rect])
fig.savefig("temp.jpg")