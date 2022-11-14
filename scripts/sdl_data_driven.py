#from helper import *
# Imports
import numpy as np
import pandas as pd
from NNetwork import NNetwork as nn
import networkx as nx
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

df_rst = []
ntwk_names = ['Caltech36', 'nws-20000-1000-05', 'UCLA26'] 
for ntwk in ntwk_names:
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
            
        for ca in ["kura", "fca", "ghm"]:
            temp = [ca, num_nodes, ntwk]
            name_dynamics = "motifDynamics/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_PARAMS-csv.pkl'
            df_dynamics = pickle.loads(s3_bucket.Object(name_dynamics).get()['Body'].read())
            #############################CHANGE########################################
            name_coladj = "motifDynamics/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_COLADJ-'+str(ca)+'_PARAMS-csv.pkl'
            df_coladj = pickle.loads(s3_bucket.Object(name_coladj).get()['Body'].read())
            #############################CHANGE########################################
            print(name_coladj+"\n")
            
            #theory driven
            if ntwk == "nws-20000-1000-05":
                sample_size = 100 #number of samples used for learning dictionary
                ind_dense = sorted(range(len(df_feature.density)), key=lambda i: df_feature.density[i], reverse=True)[:sample_size]
                ind_sparse = df_feature[df_feature.is_tree != True].sort_values(by='density').index[:sample_size].tolist()
                ind_con = df_dynamics[df_dynamics==True].index.tolist()
                if len(ind_con)>sample_size:
                    ind_con = ind_con[:sample_size]
                X_comb = pd.concat([pd.DataFrame(X.T), df_coladj/max(df_coladj.max())], axis=1)


                r = 4
                W_dense, H_dense = ALS(X=X_comb.loc[ind_dense,].T.values, 
                                       n_components=r, n_iter=100, a0 = 0, a1 = 0, a12 = 0, H_nonnegativity=True, 
                                       W_nonnegativity=True, compute_recons_error=True, subsample_ratio=1)
                W_sparse, H_sparse = ALS(X=X_comb.loc[ind_sparse,].T.values, 
                                         n_components=r, n_iter=100, a0 = 0, a1 = 0, a12 = 0, 
                                         H_nonnegativity=True, W_nonnegativity=True, 
                                         compute_recons_error=True, subsample_ratio=1)
                if ca != "ghm": 
                    dynamicstree_filename = "motifDynamics/SAMPLES-100_NTWK-tree_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_PARAMS-csv.pkl'
                    coladjtree_filename = "motifDynamics/SAMPLES-100_NTWK-tree_K-"+str(num_nodes)+'_COLADJ-'+str(ca)+'_PARAMS-csv.pkl'
                    ntwktree_filename = "motifSampling/SAMPLES-100_NTWK-tree_K-"+str(num_nodes)+"_PATCHES.pkl"
                    X_tree = pickle.loads(s3_bucket.Object(ntwktree_filename).get()['Body'].read())
                    df_dynamicstree = pickle.loads(s3_bucket.Object(dynamicstree_filename).get()['Body'].read())
                    df_coladjtree = pickle.loads(s3_bucket.Object(coladjtree_filename).get()['Body'].read())
                    X_tree_comb = pd.concat([pd.DataFrame(X_tree.T), df_coladjtree/max(df_coladjtree.max())], axis=1)

                    W_con, H_con = ALS(X=X_comb.loc[ind_con,].T.values, n_components=r, 
                                       n_iter=100, a0 = 0, a1 = 0, a12 = 0, H_nonnegativity=True, 
                                       W_nonnegativity=True, compute_recons_error=True, subsample_ratio=1)
                    W_tree, H_tree = ALS(X=X_tree_comb.T.values, n_components=r, 
                                       n_iter=100, a0 = 0, a1 = 0, a12 = 0, H_nonnegativity=True, 
                                       W_nonnegativity=True, compute_recons_error=True, subsample_ratio=1)
                    W = np.concatenate([W_dense.T, W_sparse.T, W_con.T, W_tree.T])
                else:
                    W = np.concatenate([W_dense.T, W_sparse.T])
                s3_bucket.put_object(Body=pickle.dumps(W), 
                                     Key="output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_sdl.pkl')
            
            y = df_dynamics.y
            base = df_dynamics.baseline_width
            df_dynamics = df_dynamics[[c for c in df_dynamics.columns if c.startswith('s')]]
            df_coladj = pd.concat([pd.DataFrame(X.T), df_coladj/max(df_coladj.max())], axis=1)
            
            Y_data = y
            under_sampler = RandomUnderSampler(random_state=42)
            X_res, y_res = under_sampler.fit_resample(df_coladj.values, Y_data)
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                                test_size = 0.2, 
                                                                random_state = 4, 
                                                                stratify = y_res)
            xy_dict = {}
            xy_dict["X_train"] = X_train
            xy_dict["X_test"] = X_test
            xy_dict["y_train"] = y_train
            xy_dict["y_test"] = y_test
            binary_stream = pickle.dumps(xy_dict)
            s3_bucket.put_object(Body=binary_stream, Key="sdl_xy/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'.pkl')
            
            #data-driven
            sdl_filename = "output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_sdl.pkl'
            xi = 1
            iter_avg = 1
            beta = 0.5
            iteration = 100
            r = 2
            SDL_BCD_class_new = SDL_BCD(X=[X_train.T, y_train.to_numpy().reshape(-1,1).T],  # data, label
                                    X_test=[X_test.T, y_test.to_numpy().reshape(-1,1).T],
                                    n_components=r, xi=xi, L1_reg = [0,0,0], L2_reg = [0,0,0], 
                                    nonnegativity=[True,True,False],full_dim=False)
            results_dict_new = SDL_BCD_class_new.fit(iter=iteration, subsample_size=None,
                                                            beta = beta,
                                                            search_radius_const=np.linalg.norm(X_train),
                                                            update_nuance_param=False,
                                                            if_compute_recons_error=False, if_validate=False)
            temp.append(results_dict_new['AUC'])
            temp.append(results_dict_new['Accuracy'])
            temp.append(results_dict_new['Precision'])
            temp.append(results_dict_new['Recall'])
            temp.append(results_dict_new['F_score'])
            binary_stream = pickle.dumps(results_dict_new)
            s3_bucket.put_object(Body=binary_stream, Key=sdl_filename)

            rf = RandomForestClassifier(random_state = 42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            temp.append(precision_score(y_test, y_pred))
            temp.append(recall_score(y_test, y_pred))
            temp.append(f1_score(y_test, y_pred))
            temp.append(accuracy_score(y_test, y_pred))

            Y_data = y
            under_sampler = RandomUnderSampler()
            X_res, y_res = under_sampler.fit_resample(pd.concat([df_dynamics, base], axis=1, join='inner').copy(), Y_data)
            Y_baseline = X_res.baseline_width
            Y_data = y_res
            #baseline model
            length = len(Y_baseline[Y_baseline==False])
            Y_baseline[random.sample(list(Y_baseline[Y_baseline==False].index),length//2)] = True
            temp.append(precision_score(Y_data, Y_baseline))
            temp.append(recall_score(Y_data, Y_baseline))
            temp.append(f1_score(Y_data, Y_baseline))
            temp.append(accuracy_score(Y_data, Y_baseline))
            print(temp)
            df_rst.append(temp)
            s3_bucket.put_object(Body=pickle.dumps(temp), Key="SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_base_sdl_rf_perm.pkl')