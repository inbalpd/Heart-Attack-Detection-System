from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from matplotlib import cm
from collections import defaultdict
import networkx as nx
from sklearn import cluster
import scipy.cluster.hierarchy as sch
from sklearn.cluster import MeanShift
from matplotlib import pyplot
from numpy import where
from numpy import unique
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold, datasets
from scipy import ndimage
from time import time
import graphviz
import mglearn
from IPython.display import Image
import sklearn.tree as tree
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from pandas import get_dummies
from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from sklearn.datasets import make_classification
from sklearn import tree
from scipy.stats import randint as sp_randint, uniform
import random as random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth
from numpy import savetxt

# # # ------------------------------ data ----------------------------------------
# XY_data = pd.read_csv(r'C:\Users\Tal Yachini\Downloads\Xy_train.csv')

# ########## age value treatment ###############################
# age_and_gender = XY_data[['age', 'gender']]
# normalAge_women = age_and_gender[(
#     age_and_gender.age < 120) & (age_and_gender.gender == 0)]
# Normal_women_mean = normalAge_women.age.mean()
# normalAge_men = age_and_gender[(
#     age_and_gender.age < 120) & (age_and_gender.gender == 1)]
# Normal_men_mean = normalAge_men.age.mean()

# XY_data.loc[((XY_data.age) > 120) & (
#     XY_data.gender == 0), 'age'] = Normal_women_mean

# XY_data.loc[((XY_data.age) > 120) & (
#     XY_data.gender == 1), 'age'] = Normal_men_mean

# ######### thal values treatment #############
# XY_data.loc[((XY_data.thal) == 0), 'thal'] = 2

# ############ drop id ###########
# XY_data = XY_data.drop('id', 1)

# ############# drop trestbps ###########
# XY_data = XY_data.drop('trestbps', 1)

# # ############Continuous to Categorical - age ############

# XY_data.age = pd.cut(XY_data.age, bins=[0, 10, 20, 30, 40, 50, 60, 70,
#                                         80, 90, 120], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# XY_data.rename(columns={'age': 'ageGroup'}, inplace=True)


# ############################# dummies ##############################
# # convert categorials to dummies
# data = XY_data
# # change gender to dummy
# dummy = pd.get_dummies(data['gender'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "Female", 1: "Male"})

# # change cp to dummy
# dummy = pd.get_dummies(data['cp'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={
#                    0: "cp_typical angina", 1: "cp_atypical angina", 2: "cp_non — anginal pain", 3: "cp_asymptotic"})

# # change fbs to dummy
# dummy = pd.get_dummies(data['fbs'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "fbs_under 120", 1: "fbs_over 120"})

# # change restecg to dummy
# dummy = pd.get_dummies(data['restecg'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "restecg_normal",
#                             1: "restecg_abnormal", 2: "restecg_hyperthrophy"})

# # change exang to dummy
# dummy = pd.get_dummies(data['exang'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "exang_no", 1: "exang_yes"})

# # change slope to dummy
# dummy = pd.get_dummies(data['slope'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "slope_upsloping",
#                             1: "slope_flat", 2: "slope_downsloping"})

# # change ca to dummy
# dummy = pd.get_dummies(data['ca'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "ca_0", 1: "ca_1",
#                             2: "ca_2", 3: "ca_3", 4: "absence"})

# # change thal to dummy
# dummy = pd.get_dummies(data['thal'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={1: "thal_fixed defect",
#                             2: "thal_normal", 3: "thal_reversible defect"})

# # change age to dummy
# dummy = pd.get_dummies(data['ageGroup'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={1: "age_0-10",
#                             2: "age_11-20", 3: "age_21-30", 4: "age_31-40", 5: "age_41-50",
#                             6: "age_51-60", 7: "age_61-70", 8: "age_71-80", 9: "age_81-90", 10: "age_91+"})

# # remove categorial after dummies
# data = data.drop('gender', 1)
# data = data.drop('cp', 1)
# data = data.drop('fbs', 1)
# data = data.drop('restecg', 1)
# data = data.drop('exang', 1)
# data = data.drop('slope', 1)
# data = data.drop('ca', 1)
# data = data.drop('thal', 1)
# data = data.drop('ageGroup', 1)

# # #################################### start Decision tree #################################

# # ################# separate X and Y data ##############
# X = data.drop('y', 1)
# Y = data['y']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.23, random_state=123)

# full tree ########################### 1.1

# model = DecisionTreeClassifier(random_state=66)
# model.fit(X_train, y_train)

# # plt.figure(figsize=(19, 12))
# # plot_tree(model, filled=True, class_names=True)
# # plt.show()

# print(
#     f"Train Accuracy: {accuracy_score(y_true=y_train, y_pred=model.predict(X_train)):.2f}")
# print(
#     f"Test Accuracy: {accuracy_score(y_true=y_test, y_pred=model.predict(X_test)):.2f}")

# # ####################### size check #################
# print(f"Train size: {X_train.shape[0]}")
# print(f"Test size: {X_test.shape[0]}")

# print("Train\n-----------\n", pd.value_counts(y_train)/y_train.shape[0])
# print("\nTest\n-----------\n", pd.value_counts(y_test)/y_test.shape[0])


# hyper tuning ############################################## 1.2

# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.1, random_state=123)


# # ####################### grid search ################################
# param_grid = {'max_depth': np.arange(1, 10, 1),
#               'min_samples_leaf': np.arange(1, 7, 1),
#               # 'criterion': ['entropy', 'gini'],
#               'min_samples_split': np.arange(2, 8, 1),
#               'max_features': [5, 6, 7, 8, 9, 10, 11, 12, 13]
#               }

# comb = 1
# for list_ in param_grid.values():
#     comb *= len(list_)
# print(comb)


# grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=66),
#                            param_grid=param_grid,
#                            refit=True,
#                            cv=5)

# grid_search.fit(X_train, y_train)


# best_model = grid_search.best_estimator_
# preds = best_model.predict(X_test)
# print("Test accuracy: ", round(accuracy_score(y_test, preds), 3))
# print(best_model)
# print ("Running time to fine the best DT: ", grid_search.refit_time_ , " seconds")
# # ###################### grid end #######################################

# # ###################### Random Search ##################################
# random_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=66),
#                                    param_distributions=param_grid, cv=5,
#                                    random_state=666, n_iter=2000, refit=True)


# random_search.fit(X_train, y_train)
# preds = random_search.predict(X_test)
# print("Test accuracy: ", round(accuracy_score(y_test, preds), 3))
# print("Best Model: ", random_search.best_estimator_)

# ######################## Random end ######################################

# #################### final Decision Tree #################################
# model = DecisionTreeClassifier(
#     max_depth=4, max_features=9, min_samples_split=3, random_state=66)

# model.fit(X_train, y_train)

# # plt.figure(figsize=(19, 12))

# # fn = list(X.columns)
# # cn = ['0', '1']
# # fig = tree.plot_tree(model,
# #                      feature_names=fn,
# #                      class_names=cn,
# #                      filled=True,
# #                      rounded=True)

# # pydot_graph = pydotplus.graph_from_dot_data(fig)
# # pydot_graph.write_png('original_tree.png')
# # pydot_graph.set_size('"5,5!"')
# # pydot_graph.write_png('resized_tree.png')

# # plt.show()

# preds = model.predict(X_test)
# print(
#     f"Train accuracy: {accuracy_score(y_true=y_train, y_pred=model.predict(X_train)):.3f}")
# print("Test accuracy: ", round(accuracy_score(y_test, preds), 3))

# # # ####################### final Decision Tree end ###############################

# !############################################## check one row (176) 1.3

# first_row = X_test.iloc[:1]
# pd.set_option("display.max_rows", None, "display.max_columns", None)
# print(first_row)

# importance of features
# features = X.columns
# important_features = model.feature_importances_
# indices = np.argsort(important_features)[:: 1]

# print(important_features)
# plt.figure(1)
# plt.title('Feature Importance')
# plt.barh(range(len(indices)),
#          important_features[indices], color='b', align='center')
# plt.yticks(range(len(indices)), features[indices])
# plt.xlabel('Relative Importance')
# plt.show()

########## visualiztion #############

# def GridSearch_table_plot(grid_clf, param_name,
#                           num_results=15,
#                           negative=True,
#                           graph=True,
#                           display_all_params=True):

#     from IPython.display import display

#     clf = grid_clf.best_estimator_
#     clf_params = grid_clf.best_params_
#     if negative:
#         clf_score = -grid_clf.best_score_
#     else:
#         clf_score = grid_clf.best_score_
#     clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
#     cv_results = grid_clf.cv_results_

#     print("best parameters: {}".format(clf_params))
#     print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
#     if display_all_params:
#         import pprint
#         pprint.pprint(clf.get_params())

#     # pick out the best results
#     # =========================
#     scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

#     best_row = scores_df.iloc[0, :]
#     if negative:
#         best_mean = -best_row['mean_test_score']
#     else:
#         best_mean = best_row['mean_test_score']
#     best_stdev = best_row['std_test_score']
#     best_param = best_row['param_' + param_name]

#     # display the top 'num_results' results
#     # =====================================
#     display(pd.DataFrame(cv_results)
#             .sort_values(by='rank_test_score').head(num_results))

#     # plot the results
#     # ================
#     scores_df = scores_df.sort_values(by='param_' + param_name)

#     if negative:
#         means = -scores_df['mean_test_score']
#     else:
#         means = scores_df['mean_test_score']
#     stds = scores_df['std_test_score']
#     params = scores_df['param_' + param_name]

#     # plot
#     if graph:
#         plt.figure(figsize=(8, 8))
#         plt.errorbar(params, means, yerr=stds)

#         plt.axhline(y=best_mean + best_stdev, color='red')
#         plt.axhline(y=best_mean - best_stdev, color='red')
#         plt.plot(best_param, best_mean, 'or')

#         plt.title(
#             param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
#         plt.xlabel(param_name)
#         plt.ylabel('Score')
#         plt.show()


# GridSearch_table_plot(grid_search, "max_depth", negative=False)
# GridSearch_table_plot(grid_search, "min_samples_split", negative=False)
# GridSearch_table_plot(grid_search, "min_samples_leaf", negative=False)
# GridSearch_table_plot(grid_search, "max_features", negative=False)

# ____________________________________________________ End Decision Tree _______________________________________________________


# #################### Neural Networks  - ANN #####################################
# scaler = StandardScaler()
# # StandardScaler with categorial
# X_train_s = X_train

# # calculate mean and std per each continues variable
# chol_mean = X_train_s['chol'].mean()
# chol_std = X_train_s['chol'].std()

# thalach_mean = X_train_s['thalach'].mean()
# thalach_std = X_train_s['thalach'].std()

# oldpeak_mean = X_train_s['oldpeak'].mean()
# oldpeak_std = X_train_s['oldpeak'].std()

# X_train_s[['chol', 'thalach', 'oldpeak']] = scaler.fit_transform(
#     X_train_s[['chol', 'thalach', 'oldpeak']])

# # separate to continues and categorials:
# # train:
# X_train_continues = X_train_s[['chol', 'thalach', 'oldpeak']]
# X_train_categorials = X_train_s.drop(['chol', 'thalach', 'oldpeak'], 1)
# X_train_categorials = X_train_categorials.replace({0: -1})
# X_train_s = pd.concat([X_train_continues, X_train_categorials], axis=1)


# # test:

# X_test_s = X_test

# X_test_continues = X_test_s[['chol', 'thalach', 'oldpeak']]

# # fit test set by the mean and std of the train set
# X_test_continues['chol'] = (X_test_continues['chol']-chol_mean)/chol_std
# X_test_continues['thalach'] = (
#     X_test_continues['thalach']-thalach_mean)/thalach_std
# X_test_continues['oldpeak'] = (
#     X_test_continues['oldpeak']-oldpeak_mean)/oldpeak_std

# X_test_categorials = X_test_s.drop(['chol', 'thalach', 'oldpeak'], 1)
# X_test_categorials = X_test_categorials.replace({0: -1})
# X_test_s = pd.concat([X_test_continues, X_test_categorials], axis=1)

# ###############
# model = MLPClassifier(random_state=66
#                       # , verbose=True
#                       )
# model.fit(X_train_s, y_train)

# print(
#     f"Train Accuracy: {accuracy_score(y_true=y_train, y_pred=model.predict(X_train_s)):.2f}")
# print(
#     f"Test Accuracy: {accuracy_score(y_true=y_test, y_pred=model.predict(X_test_s)):.2f}")

# plt.plot(model.loss_curve_)
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.show()

# # ###################### Random Search ##################################

# comb = 1
# for list_ in parameter_space.values():
#     comb *= len(list_)
# print(comb)
# res = pd.DataFrame()

# # -----------------hyper-tuning--------------------


# def hidden_layer():
#     hidden_layer = []
#     for i in range(1, 25):
#         temp1 = (i,)
#         hidden_layer.append(temp1)
#         for j in range(1, 25):
#             temp2 = (i, j)
#             hidden_layer.append(temp2)
#     return hidden_layer


# hidden_layer = hidden_layer()
# parameter_space = {
#     'hidden_layer_sizes': hidden_layer,
#     'max_iter': np.arange(100, 700, 10),
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': np.arange(0.00, 0.001, 0.0005),
#     'learning_rate_init':  np.arange(0.05, 15, 0.01),
#     'early_stopping': [True],
#     'learning_rate': ['constant', 'adaptive']
# }

# random_search_ANN = RandomizedSearchCV(
#     model, parameter_space, cv=5, random_state=876, n_iter=2000, refit=True)

# random_search_ANN.fit(X_train_s, y_train)
# preds = random_search_ANN.predict(X_test_s)

# print(
#     f"Train accuracy: {accuracy_score(y_true=y_train, y_pred=random_search_ANN.predict(X_train_s)):.3f}")
# print("Test accuracy: ", (accuracy_score(y_test, preds)))
# print("Best Model: ", random_search_ANN.best_estimator_)
# best_model = random_search_ANN.best_estimator_
# results = random_search_ANN.cv_results_

# print("Running time of the best ANN: ",
#       random_search_ANN.refit_time_, " seconds")
# nor = pd.DataFrame(results)
# nor.to_csv(r'C:\Users\Tal Yachini\Downloads\ANN_RandomSearch_results.csv')

# # # ######################## Random end ######################################
# # ----------------------visualization--------------------------


# def randomSearch_table_plot(grid_clf, param_name,
#                             num_results=15,
#                             negative=True,
#                             graph=True,
#                             display_all_params=True):

#     from IPython.display import display

#     clf = grid_clf.best_estimator_
#     clf_params = grid_clf.best_params_
#     if negative:
#         clf_score = -grid_clf.best_score_
#     else:
#         clf_score = grid_clf.best_score_
#     clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
#     cv_results = grid_clf.cv_results_

#     print("best parameters: {}".format(clf_params))
#     print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
#     if display_all_params:
#         import pprint
#         pprint.pprint(clf.get_params())

#     # pick out the best results
#     # =========================
#     scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

#     best_row = scores_df.iloc[0, :]
#     if negative:
#         best_mean = -best_row['mean_test_score']
#     else:
#         best_mean = best_row['mean_test_score']
#     best_stdev = best_row['std_test_score']
#     best_param = best_row['param_' + param_name]

#     # display the top 'num_results' results
#     # =====================================
#     display(pd.DataFrame(cv_results).sort_values(
#         by='rank_test_score').head(num_results))
#     #scores_df['hidden_layer_sizes'] = float('.'.join(str(ele) for ele in scores_df['hidden_layer_sizes']))
#     # plot the results
#     # ================
#     scores_df = scores_df.sort_values(by='param_' + param_name)

#     if negative:
#         means = -scores_df['mean_test_score']
#     else:
#         means = scores_df['mean_test_score']
#     stds = scores_df['std_test_score']
#     params = scores_df['param_' + param_name]

#     # plot
#     if graph:
#         plt.figure(figsize=(8, 8))
#         plt.errorbar(params, means, yerr=stds)

#         plt.axhline(y=best_mean + best_stdev, color='red')
#         plt.axhline(y=best_mean - best_stdev, color='red')
#         plt.plot(best_param, best_mean, 'or')

#         plt.title(
#             param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
#         plt.xlabel(param_name)
#         plt.ylabel('Score')
#         plt.show()


# randomSearch_table_plot(random_search_ANN, "max_iter", negative=False)
# randomSearch_table_plot(random_search_ANN, "activation", negative=False)
# randomSearch_table_plot(random_search_ANN, "solver", negative=False)
# randomSearch_table_plot(random_search_ANN, "alpha", negative=False)
# randomSearch_table_plot(
#     random_search_ANN, "learning_rate_init", negative=False)
# randomSearch_table_plot(random_search_ANN, "learning_rate", negative=False)

######## hidden_layer ##################################

# Results = pd.read_csv(
#     r'C:\Users\Tal Yachini\Downloads\ANN_RandomSearch_results_o.816_new.csv')
# Results = pd.DataFrame(
#     Results, columns=['param_hidden_layer_sizes', 'mean_test_score']).head(50)
# Results.plot(x='param_hidden_layer_sizes',
#              y='mean_test_score', kind='line', color='tomato')

# # y = pd.DataFrame(Results['mean_test_score'])
# y = Results['mean_test_score'].values.tolist()
# # x = pd.DataFrame(Results['param_hidden_layer_sizes'])
# x = Results['param_hidden_layer_sizes'].values.tolist()
# ymax = max(y)
# xpos = y.index(ymax)
# xmax = x[xpos]

# plt.annotate('local max', xy=(xmax, ymax), xytext=(xmax, ymax+5),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )

# plt.show()
# # _____________________________________ End Neural Networks - ANN _______________________________________________________

# ######################## K-means ################################

# normalize data
# scaler = StandardScaler()
# X_s = X

# X_s[['chol', 'thalach', 'oldpeak']] = scaler.fit_transform(
#     X_s[['chol', 'thalach', 'oldpeak']])

# # separate to continues and categorials:
# # train:
# X_continues = X_s[['chol', 'thalach', 'oldpeak']]
# X_categorials = X_s.drop(['chol', 'thalach', 'oldpeak'], 1)
# X_categorials = X_categorials.replace({0: -1})
# X_s = pd.concat([X_continues, X_categorials], axis=1)


# kmeans = KMeans(n_clusters=2, random_state=66)
# kmeans.fit(X_s)
# print(kmeans.fit(X_s))
# kmeans.cluster_centers_
# print(kmeans.cluster_centers_)
# kmeans.predict(X_s)
# print(kmeans.predict(X_s))

# pca for graph
# pca = PCA(n_components=2)
# pca.fit(X_s)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_.sum())


# X_pca = pca.transform(X_s)
# X_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
# X_pca['cluster'] = data['y']
# sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_pca, palette='Set2')
# plt.title('Display Y division with PCA')
# plt.show()

# Num_ones = np.count_nonzero(kmeans.predict(X_s))
# print("Num Of ones with Kmeans: ", Num_ones)

# Num_zero = len(kmeans.predict(X_s))-Num_ones
# print("Num Of Zeros with Kmeans: ", Num_zero)

# Num_ones = np.count_nonzero(Y)
# print("Num Of ones in Y: ", Num_ones)

# Num_zero = len(Y)-Num_ones
# print("Num Of Zeros in Y: ", Num_zero)

# X_pca['cluster'] = kmeans.predict(X_s)

# sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_pca,  palette='Set2')
# plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(
#     kmeans.cluster_centers_)[:, 1], marker='+', s=200, color='black')
# plt.title('Display Kmeans prediction with PCA')
# plt.show()


######### fitness ##############################
# kmeans.fit(X_s)
# assignment = kmeans.predict(X_s)
# iner = kmeans.inertia_
# sil = silhouette_score(X_s, assignment)
# dbi = davies_bouldin_score(X_s, assignment)

# print("inertia: ", iner)
# print("silhouette: ", sil)
# print("davies_bouldin: ", dbi)

################# 8 models ###############################
# dbi_list = []
# sil_list = []

# for n_clusters in tqdm(range(2, 10, 1)):
#     kmeans = KMeans(n_clusters=n_clusters, max_iter=300,
#                     n_init=10, random_state=66)
#     kmeans.fit(X_s)
#     assignment = kmeans.predict(X_s)

#     sil = silhouette_score(X_s, assignment)
#     dbi = davies_bouldin_score(X_s, assignment)

#     dbi_list.append(dbi)
#     sil_list.append(sil)


# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('Model Check')

# ax1.plot(range(2, 10, 1), sil_list, marker='o')
# ax1.set_title("Silhouette")
# ax1.set(xlabel="Number of clusters")
# # plt.show()

# ax2.plot(range(2, 10, 1), dbi_list, marker='o')
# ax2.set_title("Davies-bouldin")
# ax2.set(xlabel="Number of clusters")

# plt.show()

# kmeans = KMeans(n_clusters=2, max_iter=300,
#                 n_init=10, random_state=66)
# kmeans.fit(X_s)

###### Adjusted Rand index (accuracy) #########

# labels_true = Y
# labels_pred = kmeans.predict(X_s)

# kmeans_Score = round(metrics.adjusted_rand_score(labels_true, labels_pred), 3)
# print("kmeans_Score: ", kmeans_Score)


################### Agglomerative Clustering  (hirarcy) ###################

# linked = linkage(X_s, 'single')
# labelList = range(0, 212)
# # plt.figure(figsize=(10, 7))
# # dendrogram(linked,
# #            orientation='top',
# #            labels=labelList,
# #            distance_sort='descending',
# #            show_leaf_counts=True)
# # plt.show()

# cluster = AgglomerativeClustering(
#     n_clusters=2, affinity='euclidean', linkage='ward')
# cluster.fit_predict(X_s)

# print(cluster.labels_)

# X_pca = pca.transform(X_s)
# X_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
# X_pca['cluster'] = cluster.labels_
# sns.scatterplot(x='PC1', y='PC2', hue=cluster.labels_,
#                 data=X_pca,  palette='Set2')

# plt.title("Clustering with Agglomerative ")
# plt.show()

# kmeans = KMeans(n_clusters=2, max_iter=300,
#                 n_init=10, random_state=66)
# kmeans.fit(X_s)


# Num_ones = np.count_nonzero(kmeans.predict(X_s))
# print("Num Of ones with Kmeans: ", Num_ones)

# Num_zero = len(kmeans.predict(X_s))-Num_ones
# print("Num Of Zeros with Kmeans: ", Num_zero)

# Num_ones = np.count_nonzero(cluster.fit_predict(X_s))
# print("Num Of ones in Agglomerative: ", Num_ones)

# Num_zero = len(cluster.fit_predict(X_s))-Num_ones
# print("Num Of Zeros in Agglomerative: ", Num_zero)
# labels_pred = cluster.labels_
# Agglomerative_Score = round(
#     metrics.adjusted_rand_score(labels_true, labels_pred), 3)
# print("Agglomerative_Score: ", Agglomerative_Score)

# ____________________________________________________ End K-means _______________________________________________________

######### final model - decision tree #########################

# finalModel = DecisionTreeClassifier(
#     max_depth=4, max_features=9, min_samples_split=3, random_state=66)

# finalModel.fit(X_train, y_train)
# print(
#     f"Accuracy: {accuracy_score(y_true=y_train, y_pred=finalModel.predict(X_train)):.3f}")
# print(confusion_matrix(y_true=y_train, y_pred=finalModel.predict(X_train)))


################ final prediction (: ###################################

# X_final_test = pd.read_csv(r'C:\Users\Tal Yachini\Downloads\X_test.csv')

# ########## age value treatment ###############################
# age_and_gender = X_final_test[['age', 'gender']]
# normalAge_women = age_and_gender[(
#     age_and_gender.age < 120) & (age_and_gender.gender == 0)]
# Normal_women_mean = normalAge_women.age.mean()
# normalAge_men = age_and_gender[(
#     age_and_gender.age < 120) & (age_and_gender.gender == 1)]
# Normal_men_mean = normalAge_men.age.mean()

# X_final_test.loc[((X_final_test.age) > 120) & (
#     X_final_test.gender == 0), 'age'] = Normal_women_mean

# X_final_test.loc[((X_final_test.age) > 120) & (
#     X_final_test.gender == 1), 'age'] = Normal_men_mean

# ######### thal values treatment #############
# X_final_test.loc[((X_final_test.thal) == 0), 'thal'] = 2

# ############ drop id ###########
# X_final_test = X_final_test.drop('id', 1)

# ############# drop trestbps ###########
# X_final_test = X_final_test.drop('trestbps', 1)

# # ############Continuous to Categorical - age ############

# X_final_test.age = pd.cut(X_final_test.age, bins=[0, 10, 20, 30, 40, 50, 60, 70,
#                                                   80, 90, 120], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# X_final_test.rename(columns={'age': 'ageGroup'}, inplace=True)


# ############################# dummies ##############################
# # convert categorials to dummies
# data = X_final_test
# # change gender to dummy
# dummy = pd.get_dummies(data['gender'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "Female", 1: "Male"})

# # change cp to dummy
# dummy = pd.get_dummies(data['cp'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={
#                    0: "cp_typical angina", 1: "cp_atypical angina", 2: "cp_non — anginal pain", 3: "cp_asymptotic"})

# # change fbs to dummy
# dummy = pd.get_dummies(data['fbs'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "fbs_under 120", 1: "fbs_over 120"})

# # change restecg to dummy
# dummy = pd.get_dummies(data['restecg'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "restecg_normal",
#                             1: "restecg_abnormal", 2: "restecg_hyperthrophy"})

# # change exang to dummy
# dummy = pd.get_dummies(data['exang'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "exang_no", 1: "exang_yes"})

# # change slope to dummy
# dummy = pd.get_dummies(data['slope'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "slope_upsloping",
#                             1: "slope_flat", 2: "slope_downsloping"})

# # change ca to dummy
# dummy = pd.get_dummies(data['ca'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={0: "ca_0", 1: "ca_1",
#                             2: "ca_2", 3: "ca_3", 4: "absence"})

# # change thal to dummy
# dummy = pd.get_dummies(data['thal'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={1: "thal_fixed defect",
#                             2: "thal_normal", 3: "thal_reversible defect"})

# # change age to dummy
# dummy = pd.get_dummies(data['ageGroup'])
# data = pd.concat([data, dummy], axis=1)
# data = data.rename(columns={1: "age_0-10",
#                             2: "age_11-20", 3: "age_21-30", 4: "age_31-40", 5: "age_41-50",
#                             6: "age_51-60", 7: "age_61-70", 8: "age_71-80", 9: "age_81-90", 10: "age_91+"})

# # remove categorial after dummies
# data = data.drop('gender', 1)
# data = data.drop('cp', 1)
# data = data.drop('fbs', 1)
# data = data.drop('restecg', 1)
# data = data.drop('exang', 1)
# data = data.drop('slope', 1)
# data = data.drop('ca', 1)
# data = data.drop('thal', 1)
# data = data.drop('ageGroup', 1)


# finalModel = DecisionTreeClassifier(
#     max_depth=4, max_features=9, min_samples_split=3, random_state=66)

# finalModel = finalModel.fit(X, Y)

# preds = finalModel.predict(data)
# preds = pd.DataFrame(preds)
# preds = preds.rename(columns={0: "y"})
# # print(preds)
# preds.to_csv(
#     r'C:\Users\Tal Yachini\Downloads\final_prediction.csv', index=False)
