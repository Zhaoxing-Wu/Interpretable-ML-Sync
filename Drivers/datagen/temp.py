import datagen
import pickle

import pickle, csv
import numpy as np
import pandas as pd
import networkx as nx
import statistics as s
from math import floor
from tqdm import trange
from NNetwork import NNetwork as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

df_graph = pickle.load(open('/Users/zhaoxingwu/Desktop/REU/fall/Interpretable-ML-Sync/data/random_graph/ucla_20walk_graph.pkl', 'rb'))
df_features = pd.read_csv("/Users/zhaoxingwu/Desktop/REU/fall/Interpretable-ML-Sync/data/random_graph/ucla_20walk_graph_features.csv")
df_g2v = pd.read_csv("/Users/zhaoxingwu/Desktop/REU/fall/Interpretable-ML-Sync/data/random_graph/ucla_20walk_graph2vec.csv", header=None)
df_n2v = pd.read_csv("/Users/zhaoxingwu/Desktop/REU/fall/Interpretable-ML-Sync/data/random_graph/ucla_20walk_node2vec.csv", header=None)
df_spec = pd.read_csv("/Users/zhaoxingwu/Desktop/REU/fall/Interpretable-ML-Sync/data/random_graph/ucla_20walk_spectral.csv", header=None)

df_dynamics = pd.read_csv("/Users/zhaoxingwu/Desktop/REU/fall/Interpretable-ML-Sync/data/random_graph/ucla_20walk_ghm.csv")
Y_data = df_dynamics.y #concentration
df_dynamics = df_dynamics.loc[:,df_dynamics.columns.str.contains("s")].copy()

##############baseline model###################
df = df_dynamics.copy()

under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = under_sampler.fit_resample(df, Y_data)
Y_baseline = X_res.baseline_width

length = len(Y_baseline[Y_baseline==False])
Y_baseline[random.sample(list(Y_baseline[Y_baseline==False].index),length//2)] = True
conf_matrix_baseline = confusion_matrix(y_true=y_res, y_pred=Y_baseline)
print("===============================baseline")
print('Precision: %.3f' % precision_score(y_res, Y_baseline))
print('Recall: %.3f' % recall_score(y_res, Y_baseline))
print('F1: %.3f' % f1_score(y_res, Y_baseline))
print('Accuracy: %.3f' % accuracy_score(y_res, Y_baseline))

###############dynamics############################
df = df_dynamics.copy()
under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = under_sampler.fit_resample(df, Y_data)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size = 0.2, 
                                                    random_state = 4, 
                                                    stratify = y_res)
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
conf_matrix_dynamics = confusion_matrix(y_true=y_test, y_pred=y_pred)
print("===============================dynamics")
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

###############dynamics+graphfeatures############################
df = pd.concat([df_dynamics, 
                df_features], axis=1, join='inner').copy()
under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = under_sampler.fit_resample(df, Y_data)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size = 0.2, 
                                                    random_state = 4, 
                                                    stratify = y_res)
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
conf_matrix_features = confusion_matrix(y_true=y_test, y_pred=y_pred)
print("===============================dynamics+graphfeatures")
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
acc_features = accuracy_score(y_test, y_pred)


###############dynamics+n2v############################
df = pd.concat([df_dynamics, 
                df_n2v], axis=1, join='inner').copy()
under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = under_sampler.fit_resample(df, Y_data)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size = 0.2, 
                                                    random_state = 4, 
                                                    stratify = y_res)
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
conf_matrix_n2v = confusion_matrix(y_true=y_test, y_pred=y_pred)
print("===============================dynamics+n2v")
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
acc_n2v = accuracy_score(y_test, y_pred)

###############dynamics+spec############################
df = pd.concat([df_dynamics, 
                df_spec], axis=1, join='inner').copy()
under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = under_sampler.fit_resample(df, Y_data)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size = 0.2, 
                                                    random_state = 4, 
                                                    stratify = y_res)
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
conf_matrix_g2v = confusion_matrix(y_true=y_test, y_pred=y_pred)
print("===============================dynamics+spec")
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
acc_g2v = accuracy_score(y_test, y_pred)

###############dynamics+g2v############################
df = pd.concat([df_dynamics, 
                df_g2v], axis=1, join='inner').copy()
under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = under_sampler.fit_resample(df, Y_data)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size = 0.2, 
                                                    random_state = 4, 
                                                    stratify = y_res)
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
conf_matrix_g2v = confusion_matrix(y_true=y_test, y_pred=y_pred)
print("===============================dynamics+g2v")
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
acc_g2v = accuracy_score(y_test, y_pred)


###############dynamics+adj############################
df = pd.concat([df_dynamics, 
                pd.DataFrame(df_graph.T)], axis=1, join='inner').copy()
under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = under_sampler.fit_resample(df, Y_data)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size = 0.2, 
                                                    random_state = 4, 
                                                    stratify = y_res)
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
conf_matrix_adj = confusion_matrix(y_true=y_test, y_pred=y_pred)
print("===============================dynamics+adj")
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
acc_adj = accuracy_score(y_test, y_pred)
