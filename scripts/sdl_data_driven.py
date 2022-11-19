from helper import *
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
import statsmodels.api as sm
from firthlogist import FirthLogisticRegression

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
            
        for ca in ["kura", "fca", "ghm"]:
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
                ind_dense = df_feature[df_dynamics.y == True].sort_values(by='density').index[-sample_size:].tolist()
                ind_sparse = df_feature[df_dynamics.y == False][df_feature.is_tree != True].sort_values(by='density').index[:sample_size].tolist()   
                ind_con = df_dynamics[df_dynamics.baseline_width==True].index.tolist()
                if len(ind_con)>sample_size:
                    ind_con = ind_con[:sample_size]
                    
                X_comb = pd.concat([pd.DataFrame(X.T), df_coladj/max(df_coladj.max())], axis=1)


                r = 1
                W_dense, H_dense = ALS(X=X_comb.loc[ind_dense,].T.values, 
                                       n_components=r, n_iter=100, a0 = 0, a1 = 0, a12 = 0, H_nonnegativity=True, 
                                       W_nonnegativity=True, compute_recons_error=True, subsample_ratio=1)
                W_sparse, H_sparse = ALS(X=X_comb.loc[ind_sparse,].T.values, 
                                         n_components=r, n_iter=100, a0 = 0, a1 = 0, a12 = 0, 
                                         H_nonnegativity=True, W_nonnegativity=True, 
                                         compute_recons_error=True, subsample_ratio=1)
                
                data_dict = {}
                data_dict["x_dense"] = X_comb.loc[ind_dense,]
                data_dict["y_dense"] = df_dynamics.y[ind_dense]
                data_dict["x_sparse"] = X_comb.loc[ind_sparse,]
                data_dict["y_sparse"] = df_dynamics.y[ind_sparse]
                
                if ca != "ghm": 
                    data_dict["x_concentrated"] = X_comb.loc[ind_con,]
                    data_dict["y_concentrated"] = df_dynamics.y[ind_con]
                    
                    dynamicstree_filename = "motifDynamics/SAMPLES-100_NTWK-tree_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_PARAMS-csv.pkl'
                    coladjtree_filename = "motifDynamics/SAMPLES-100_NTWK-tree_K-"+str(num_nodes)+'_COLADJ-'+str(ca)+'_PARAMS-csv.pkl'
                    ntwktree_filename = "motifSampling/SAMPLES-100_NTWK-tree_K-"+str(num_nodes)+"_PATCHES.pkl"
                    X_tree = pickle.loads(s3_bucket.Object(ntwktree_filename).get()['Body'].read())
                    df_dynamicstree = pickle.loads(s3_bucket.Object(dynamicstree_filename).get()['Body'].read())
                    df_coladjtree = pickle.loads(s3_bucket.Object(coladjtree_filename).get()['Body'].read())
                    X_tree_comb = pd.concat([pd.DataFrame(X_tree.T), df_coladjtree/max(df_coladjtree.max())], axis=1)[df_dynamicstree.y==True]

                    W_con, H_con = ALS(X=X_comb.loc[ind_con,].T.values, n_components=r, 
                                       n_iter=100, a0 = 0, a1 = 0, a12 = 0, H_nonnegativity=True, 
                                       W_nonnegativity=True, compute_recons_error=True, subsample_ratio=1)
                    W_tree, H_tree = ALS(X=X_tree_comb.T.values, n_components=r, 
                                       n_iter=100, a0 = 0, a1 = 0, a12 = 0, H_nonnegativity=True, 
                                       W_nonnegativity=True, compute_recons_error=True, subsample_ratio=1)
                    
                    data_dict["x_tree"] = X_tree_comb
                    data_dict["y_tree"] = df_dynamicstree.y[df_dynamicstree.y==True]
                    
                    W = np.concatenate([W_dense.T, W_sparse.T, W_con.T, W_tree.T])
                else:
                    W = np.concatenate([W_dense.T, W_sparse.T])
                    
                s3_bucket.put_object(Body=pickle.dumps(W), 
                                     Key="output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_sdl-r1.pkl')
                
                s3_bucket.put_object(Body=pickle.dumps(data_dict), 
                                     Key="output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_sdl-examples-dict.pkl')
                
                Y_data = df_dynamics.y
                under_sampler = RandomUnderSampler()
                X_res, y_res = under_sampler.fit_resample(X_comb.values, Y_data)
                X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size = 0.2, 
                                                    random_state = 4, 
                                                    stratify = y_res)
                
                                
                model = FirthLogisticRegression(wald=True)
                model.fit(np.matmul(W, X_train.T).T, y_train)
                print(model.summary())
                print(accuracy_score(y_test, model.predict(np.matmul(W, X_test.T).T).round()))
                logreg = {"summary": model.summary(),
                         "predict_prob": model.predict(np.matmul(W, X_test.T).T)}
                s3_bucket.put_object(Body=pickle.dumps(logreg), 
                                     Key = "output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_logreg-r1.pkl')
                    
                xi = 1
                iter_avg = 1
                beta = 0.5
                iteration = 100
                r = 4
                if ca == "ghm":
                    SDL_BCD_class_new = SDL_BCD(X=[np.concatenate([data_dict["x_dense"],
                                                                   data_dict["x_sparse"]]).T,
                                                np.concatenate([data_dict["y_dense"],
                                                               data_dict["y_sparse"]]).reshape(-1,1).T],  # data, label
                                        X_test=[X_test.T, y_test.to_numpy().reshape(-1,1).T],
                                        n_components=r, xi=xi, L1_reg = [0,0,0], L2_reg = [0,0,0], 
                                        nonnegativity=[True,True,False],full_dim=False)
                    results_dict_new = SDL_BCD_class_new.fit(iter=iteration, subsample_size=None,
                                                                beta = beta, search_radius_const = np.linalg.norm(X_train), update_nuance_param=False, if_compute_recons_error=False, if_validate=False)
                    print("Theory driven SDL"+ str(results_dict_new["Accuracy"]))
                    s3_bucket.put_object(Body=pickle.dumps(results_dict_new),
                                         Key = "output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_sdlsdl-r4.pkl')
                
                else:
                    SDL_BCD_class_new = SDL_BCD(X=[np.concatenate([data_dict["x_dense"],
                                                                   data_dict["x_sparse"], 
                                                                   data_dict["x_concentrated"], 
                                                                   data_dict["x_tree"]]).T,
                                                np.concatenate([data_dict["y_dense"],
                                                               data_dict["y_sparse"], 
                                                               data_dict["y_concentrated"], 
                                                               data_dict["y_tree"]]).reshape(-1,1).T],  # data, label
                                        X_test=[X_test.T, y_test.to_numpy().reshape(-1,1).T],
                                        n_components=r, xi=xi, L1_reg = [0,0,0], L2_reg = [0,0,0], 
                                        nonnegativity=[True,True,False],full_dim=False)
                    results_dict_new = SDL_BCD_class_new.fit(iter=iteration, subsample_size=None,
                                                                beta = beta, search_radius_const = np.linalg.norm(X_train), update_nuance_param=False, if_compute_recons_error=False, if_validate=False)
                    print("Theory driven SDL"+ str(results_dict_new["Accuracy"]))
                    s3_bucket.put_object(Body=pickle.dumps(results_dict_new),
                                         Key = "output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_sdlsdl-r4.pkl')

###########################################       
ntwk_names = ['nws-20000-1000-05', 'Caltech36', 'UCLA26'] 
for ntwk in ntwk_names:
    for num_nodes in [10, 15, 20, 25, 30]:
        for ca in ["kura", "fca", "ghm"]:
            #data driven sdl
            xy = pickle.loads(s3_bucket.Object("sdl_xy/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+".pkl").get()['Body'].read())
            
            X_train = xy["X_train"]
            y_train = xy["y_train"]
            X_test = xy["X_test"]
            y_test = xy["y_test"]
            xi = 1
            iter_avg = 1
            beta = 0.5
            iteration = 100
            r = 4
            SDL_BCD_class_new = SDL_BCD(X=[X_train.T, y_train.to_numpy().reshape(-1,1).T],  # data, label
                                    X_test=[X_test.T, y_test.to_numpy().reshape(-1,1).T],
                                    n_components=r, xi=xi, L1_reg = [0,0,0], L2_reg = [0,0,0], 
                                    nonnegativity=[True,True,False],full_dim=False)
            results_dict_new = SDL_BCD_class_new.fit(iter=iteration, subsample_size=None,
                                                            beta = beta,
                                                            search_radius_const=np.linalg.norm(X_train),
                                                            update_nuance_param=False,
                                                            if_compute_recons_error=False, if_validate=False)
            print(results_dict_new["Accuracy"])
            s3_bucket.put_object(Body= pickle.dumps(results_dict_new), 
                                 Key="output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_sdl-r4.pkl')
            
            if ntwk != 'nws-20000-1000-05':
                #theory driven nmf+logreg
                W = pickle.loads(s3_bucket.Object("output/SAMPLES-10000_NTWK-nws-20000-1000-05_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_sdl-r1.pkl').get()['Body'].read())
                model = FirthLogisticRegression(wald=True)
                model.fit(np.matmul(W, X_train.T).T, y_train)
                print(model.summary())
                print(accuracy_score(y_test, model.predict(np.matmul(W, X_test.T).T).round()))
                logreg = {"summary": model.summary(),
                         "predict_prob": model.predict(np.matmul(W, X_test.T).T)}
                s3_bucket.put_object(Body=pickle.dumps(logreg), 
                                     Key = "output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_logreg-r1.pkl')
                
                #theory driven sdl
                data_dict = pickle.loads(s3_bucket.Object("output/SAMPLES-10000_NTWK-nws-20000-1000-05_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_sdl-examples-dict.pkl').get()['Body'].read())
                
                xi = 1
                iter_avg = 1
                beta = 0.5
                iteration = 100
                r = 4
                if ca == "ghm":
                    SDL_BCD_class_new = SDL_BCD(X=[np.concatenate([data_dict["x_dense"],
                                                                   data_dict["x_sparse"]]).T,
                                                np.concatenate([data_dict["y_dense"],
                                                               data_dict["y_sparse"]]).reshape(-1,1).T],  # data, label
                                        X_test=[X_test.T, y_test.to_numpy().reshape(-1,1).T],
                                        n_components=r, xi=xi, L1_reg = [0,0,0], L2_reg = [0,0,0], 
                                        nonnegativity=[True,True,False],full_dim=False)
                    results_dict_new = SDL_BCD_class_new.fit(iter=iteration, subsample_size=None,
                                                                beta = beta, search_radius_const = np.linalg.norm(X_train), update_nuance_param=False, if_compute_recons_error=False, if_validate=False)
                    print("Theory driven SDL"+ str(results_dict_new["Accuracy"]))
                    s3_bucket.put_object(Body=pickle.dumps(results_dict_new),
                                         Key = "output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_sdlsdl-r4.pkl')
                
                else:
                    SDL_BCD_class_new = SDL_BCD(X=[np.concatenate([data_dict["x_dense"],
                                                                   data_dict["x_sparse"], 
                                                                   data_dict["x_concentrated"], 
                                                                   data_dict["x_tree"]]).T,
                                                np.concatenate([data_dict["y_dense"],
                                                               data_dict["y_sparse"], 
                                                               data_dict["y_concentrated"], 
                                                               data_dict["y_tree"]]).reshape(-1,1).T],  # data, label
                                        X_test=[X_test.T, y_test.to_numpy().reshape(-1,1).T],
                                        n_components=r, xi=xi, L1_reg = [0,0,0], L2_reg = [0,0,0], 
                                        nonnegativity=[True,True,False],full_dim=False)
                    results_dict_new = SDL_BCD_class_new.fit(iter=iteration, subsample_size=None,
                                                                beta = beta, search_radius_const = np.linalg.norm(X_train), update_nuance_param=False, if_compute_recons_error=False, if_validate=False)
                    print("Theory driven SDL"+ str(results_dict_new["Accuracy"]))
                    s3_bucket.put_object(Body=pickle.dumps(results_dict_new),
                                         Key = "output/SAMPLES-10000_NTWK-"+ntwk+"_K-"+str(num_nodes)+'_DYNAMIC-'+str(ca)+'_theory_driven_sdlsdl-r4.pkl')