from erroranalysis._internal.error_analyzer import ModelAnalyzer
from erroranalysis._internal.cohort_filter import filter_from_cohort
from erroranalysis._internal.surrogate_error_tree import (compute_error_tree)
from erroranalysis._internal.constants import ModelTask
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

import numpy as np
import csv
import pandas as pd
import sys


def proximityMatrix(model, X, n_train, n_test):
    print("Calculating proximity matrix...")

    terminals = model.apply(X)

    n_total = n_train + n_test
    prox_matrix = np.empty(shape=(n_total, n_total))

    for i in range(n_total):
        for j in range(n_total):
            prox_matrix[i,j] = np.sum(np.equal(terminals[i,:], terminals[j,:]))

    nTrees = terminals.shape[1]
    prox_matrix = prox_matrix / nTrees

    for i in range(n_train):
      for j in range(n_total):
        prox_matrix[i][j] = 0

    for i in range(n_test):
      for j in range(n_test):
        prox_matrix[i + n_train][j + n_train] = 0

    X_indices = X.index.values.tolist()

    df_prox_mat = pd.DataFrame(prox_matrix, index=X_indices, columns=X_indices)

    return df_prox_mat 


def app_Domain_Prox(prox_matrix, cutoff):
  sum = 0
  index_list = []

  for index, row in prox_matrix.iterrows():
    for col_name in prox_matrix.columns:
      if prox_matrix.loc[index][col_name] > cutoff :
        sum = sum + 1
        index_list.append(index)
        break

  return index_list, sum


def app_Domain_Leverage(training_matrix, test_matrix):

  X_train_indices = training_matrix.index.values.tolist()
  X_test_indices = test_matrix.index.values.tolist()

  scaler = StandardScaler()
  scaler.fit(training_matrix)
  training_matrix = scaler.transform(training_matrix)
  test_matrix = scaler.transform(test_matrix)

  training_matrix = pd.DataFrame(training_matrix, index=X_train_indices)
  test_matrix = pd.DataFrame(test_matrix, index=X_test_indices)

  sample_num = np.shape(training_matrix)[0]
  variable_num = np.shape(training_matrix)[1]
  threshold = (3*(variable_num+1))/ sample_num

  data_frame_std_transpose = np.transpose(training_matrix)
  inverse_matrix = np.linalg.inv(np.dot(data_frame_std_transpose,training_matrix))

  test_transpose = np.transpose(test_matrix).astype(float)
  leverage = np.dot(np.dot(test_matrix, inverse_matrix),test_transpose)

  test_matrix_indices = test_matrix.index.values.tolist()

  df_leverage = pd.DataFrame(leverage, index=test_matrix_indices)
  df_leverage =  df_leverage[df_leverage.T[(df_leverage.T > threshold)].any()]

  sum = df_leverage.shape[0]
  index_list = df_leverage.index.values.tolist()


  return index_list, sum


def app_Domain_Kernel(training_matrix, test_matrix):

  X_train_indices = training_matrix.index.values.tolist()
  X_test_indices = test_matrix.index.values.tolist()

  scaler = StandardScaler()
  scaler.fit(training_matrix)
  training_matrix = scaler.transform(training_matrix)
  test_matrix = scaler.transform(test_matrix)

  training_matrix = pd.DataFrame(training_matrix, index=X_train_indices)
  test_matrix = pd.DataFrame(test_matrix, index=X_test_indices)

  kernel_den_estim = KernelDensity(kernel='gaussian').fit(training_matrix)
  log_density_array = kernel_den_estim.score_samples(training_matrix)
  threshold = (np.amin(log_density_array) + np.amax(log_density_array))/2

  kernel_den_estim_test = KernelDensity(kernel='gaussian').fit(test_matrix)
  log_density_array_test = kernel_den_estim_test.score_samples(test_matrix)

  test_matrix_indices = test_matrix.index.values.tolist()

  df_log_density = pd.DataFrame(log_density_array_test, index=test_matrix_indices)
  df_log_density =  df_log_density[df_log_density.T[(df_log_density.T > threshold)].any()]

  sum = df_log_density.shape[0]
  index_list = df_log_density.index.values.tolist()

  return index_list, sum

def app_Domain(method,X_train, X_test, clf = RandomForestClassifier(), cutoff_value=0.7):
  index_list_AD = []
  sum = 0

  if method == "proximity":
    X_merged = pd.concat([X_train, X_test], sort=False,ignore_index=False)
    n_train = len(X_train.index)
    n_test = len(X_test.index)
    prox_matrix = proximityMatrix(clf,X_merged, n_train, n_test)
    index_list_AD, sum = app_Domain_Prox(prox_matrix, cutoff_value)
  elif method == "leverage":
    index_list_AD, sum = app_Domain_Leverage(X_train, X_test)
  elif method == "KDE":
    index_list_AD, sum = app_Domain_Kernel(X_train, X_test)

  return index_list_AD, sum


def extract_errortree(error_report):
  id_metricValue = {}
  for feature in error_report:
    id_num=0
    metricValue_num=0
    for key, value in feature.items():
      if key == "id":
        id_num = value
      if key == "metricValue":
        metricValue_num = value

    #root doesn't have arg condition and represents global error, nodeName is None
    if id_num != 0:
      id_metricValue[id_num] = metricValue_num

  return id_metricValue


def traverse_tree (error_report, id_num, error):

  error_string = "Number of misclassifications : " + str('{:.0f}'.format(error)) + " | "

  parent_node = None
  condition_str = ""
  dict_filter = {}
  list_filter = []
  id_num_next=-1
  string_recursive = ""

  for feature in error_report:
    if (feature["id"] == id_num):
      for key, value in feature.items():
        if key == "arg":
          dict_filter["arg"] = [value] if value != None else []
        if key == "condition" and value != None:
          condition_str = value
          condition_all = condition_str
        if key == "parentId":
          id_num_next = value
        if key == "method" and value!= None:
          dict_filter["method"] = value
        if key == "parentNodeName" and value != None:
          parent_node = value
          dict_filter["column"] = value

      if None not in list(dict_filter.values()):
        list_filter.append(dict_filter)

  return recursive_traverse(error_report, parent_node, error_string, list_filter, condition_all, id_num_next,string_recursive)


def recursive_traverse(error_report, tree_node, error_string, list_filter, condition_all, id_num_next, string_concat):
  # if tree_node is root
  if tree_node == None or id_num_next == 0:
    string_all = error_string + (condition_all if string_concat == "" else string_concat)
    return string_all, list_filter, condition_all
  else:
    new_parent_node = ""
    condition_str = ""
    dict_filter = {}
    
    for feature in error_report:
      if (feature["id"] == id_num_next):
        for key, value in feature.items():
          if key == "arg":
            dict_filter["arg"] = [round(value,2)] if value != None else []
          if key == "condition" and value != None:
              condition_str = value
              string_concat = str(condition_str) + " ---> " + condition_all  
              condition_all = str(condition_str) + " ---> " + condition_all
          if key == "method":
            dict_filter["method"] = value
          if key == "parentId":
            id_num_next = value
          if key == "parentNodeName"  and value != None:
            new_parent_node = value
            dict_filter["column"] = value

        if None not in list(dict_filter.values()):
          list_filter.append(dict_filter)

    return recursive_traverse(error_report,new_parent_node, error_string, list_filter, condition_all, id_num_next,string_concat)


def error_filter_sort(id_metricValue, min_error):
  error_dict = {}
  str_result_cond_all = {}
  list_filter_cond_all = {}

  for key, value in id_metricValue.items():
    #filter out error values/error rates = 0, sort based on error rates
    #condition_all : description of tree traversal
    #list_filter : hierarchical list of dict objects, where each dict defines a decision rule
    string_result, list_filter, condition_all = traverse_tree (error_report, key, value)

    error_value = float(string_result.split("|")[0].split(":")[1])
    if error_value > min_error:
      str_result_cond_all[condition_all] = string_result
      error_dict[condition_all] = error_value
      list_filter_cond_all[condition_all] = list_filter
      #print(string_result)

  error_dict = dict(sorted(error_dict.items(), key=lambda item : item[1], reverse=True))

  return error_dict, str_result_cond_all, list_filter_cond_all


if __name__ == '__main__':

  ############### INSTRUCTIONS ########################################################################################################
  # To execute this work flow:
  # 1) Load Datasets: all the datasets are provided as CSV files. For Androgen Receptor Bioactivity models 
  #                   (i.e. binding, agonist, antagonist), the training and evaluation dataset files are loaded 
  #                   separately. The membrane permeability models (M11 & M12) use the same csv file 
  #                   (i.e. database_2019JCIM_M12.csv)
  # 2) Provide Configuration Parameters: 
  #                    for the app_Domain(method, X_train, X_test, clf = RandomForestClassifier(), cutoff_value=0.7) function,
  #                    the following parameters should be configured: a) method, the available options are: 
  #                    "proximity" (i.e. for the proximity matrix analysis), "leverage" and "KDE" (i.e. Kernel Density Estimation),
  #                    b) clf : the employed model, c) cutoff_value - value used to determine if instance is within or outside AD.
  #                    The clf and cutoff_value parameters are only applicable to the proximity method, default values are provided.  
  #                     
  #####################################################################################################################################                   

  ############### LOAD DATASETS #######################################################################################################
  ####  1. J. Chem. Inf. Model. 2019, 59, 2442−2455 ###################################################################################
  ####  2. Chemosphere 2021, 262, 128313            ###################################################################################
  #####################################################################################################################################

  ##### 1. Membrane Permeability  ##################################################
  Data_all = pd.read_csv('database_2019JCIM_M12.csv', index_col=0)

  X_train = Data_all[Data_all['Split'] == 'T']
  X_train = X_train.iloc[: , :-1]
  y_train = X_train.iloc[:,-1:]
  y_train = y_train.squeeze()
  X_train = X_train.iloc[:, :-1]

  X_test = Data_all[Data_all['Split'] == 'Ext']
  X_test = X_test.iloc[: , :-1]
  y_test = X_test.iloc[:,-1:]
  y_test = y_test.squeeze()
  X_test = X_test.iloc[:, :-1]

  #### 2. Androgen Receptor Bioactivity ##########################################
  #X_train = pd.read_csv('antagonist_training.csv', index_col=0)
  #y_train = X_train.iloc[:,-1:]
  #y_train = y_train.squeeze()
  #X_train = X_train.iloc[:, :-1]

  #X_test = pd.read_csv('antagonist_evaluation.csv',index_col=0)
  #y_test = X_test.iloc[:,-1:]
  #y_test = y_test.squeeze()
  #X_test = X_test.iloc[:, :-1]


  ######## CONFIGURATION ##############################################################################################################

  #### 1) classification model ##############################################
  #clf = RandomForestClassifier(n_estimators=501, bootstrap=True, max_features='sqrt')
  clf = LogisticRegression()

  #### 2) output in txt file format #########################################
  model_statistics_filename = "M12_PAMPA_Ext_results.txt"

  #### 3) minimum number of misclassifications in cohort ####################
  min_error = 5

  #### 4) proximity matrix analysis cutoff for applicability domain #########
  cutoff_value = 0.7

  #### 5) list and sum of instances in applicability domain #################
  index_list_AD, sum = app_Domain("KDE", X_train, X_test)
  #index_list_AD, sum = app_Domain("proximity",X_train, X_test,clf,0.7)


  ######## ERROR ANALYSIS   ##########################################################################################################
 
  model = clf.fit(X_train, y_train)
  y_test_pred = clf.predict(X_test)

  print("Model accuracy score: " + str(round(accuracy_score(y_test,y_test_pred),3)))

  feature_names = X_train.columns.values
  categorical_features = []
  model_task = ModelTask.CLASSIFICATION
  analyzer = ModelAnalyzer(model, X_test, y_test, feature_names, categorical_features, model_task=model_task)
  error_report, total_error, index_diff, accuracy_surrogate = analyzer.compute_error_tree(feature_names, None, None)

  id_metricValue = extract_errortree(error_report)
  error_dict, str_result_cond_all, list_filter_cond_all = error_filter_sort(id_metricValue, min_error)

  ####### APPLICABILITY DOMAIN ########################################################################################################

  with open (model_statistics_filename, 'w') as f:
    num_within_AD = "Instances within AD : " + str(sum)
    acc_model = "\nModel accuracy score: " + str(round(accuracy_score(y_test,y_test_pred),3))
    acc_surrogate = "\nSurrogate accuracy Score : " + str(accuracy_surrogate)
    total_misclassifications = "\nTotal misclassifications : " + str(len(index_diff))
    f.writelines ([num_within_AD, acc_model,acc_surrogate,total_misclassifications])

  print("Instances within AD : " + str(sum))
  print("Model accuracy score: " + str(round(accuracy_score(y_test,y_test_pred),3)))
  print("Surrogate accuracy Score : " + str(accuracy_surrogate))
  print("Total misclassifications : " + str(len(index_diff)))
  print("*******************************************************************")

  ####### PROCESS MISCLASSIFIED INSTANCES & WRITE FILTERED DATASETS ##########################################
  for key, value in error_dict.items():
    percent_error = round((value/total_error)*100)
    list_filename = model_statistics_filename.split(".txt")
    filename = list_filename[0] + "_" + str(percent_error) + "percent.csv"
    #filename = "antagonist_evaluation_" + str(percent_error) + "percent.csv"

    description = str_result_cond_all.get(key)

    with open(filename,'w') as f :
      description_print = description + "\nPercentage Error: " + str(percent_error) + " %"
      description = "{0:80} ".format(description) + str(percent_error) + " %\n"
      print(description_print)
      f.write(description)

    #dataframe output
    df = filter_from_cohort (analyzer,list_filter_cond_all[key],None)
    df_diff = df[df['Index'].isin(index_diff)]

    #indices of misclassified instances in filtered cohort
    indices = df.index.values.tolist()
    error_list = list(set(indices).intersection(set(df_diff.index.tolist())))

    #number of misclassified instances within AD
    intersection_list = list(set(error_list).intersection(set(index_list_AD)))
 
    with open(filename,'a') as f :
      f.write("Instances misclassified but within AD: " + str(len(intersection_list)) + "\n")

    print("Instances misclassified but within AD: " + str(len(intersection_list)))
    df.to_csv(filename, mode ='a')
    #print(filter_from_cohort (analyzer,listfilter_conditionall[key],None))
  
    print("---------------------------------------------------------------------\n")
