# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 16:18:50 2018

@author: user
"""

import sys
import csv
import os
import collections
import time
import numpy as np
import pickle
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF


from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score, recall_score

import ggplot



np.random.seed(46)

stime = time.time()






data_path = './dataset'


train_visit_set = set()
test_visit_set = set()

trip_set = set()
visit_set = set()
weekday_set = set()
upc_set = set()
dep_set = set()
fline_set = set()

visit_upc_scan_counter = collections.Counter()
visit_dep_scan_counter = collections.Counter()
visit_fline_scan_counter = collections.Counter()

visit2weekday = {}
visit2trip = {}



trip_upc_counters = {}
trip_dep_counters = {}
trip_fline_counters = {}



print('\n')
print('load train')
f = open(os.path.join(data_path, 'train.csv'))

counter = 0

for line in csv.reader(f):
    
    counter += 1
    
    if counter == 1:
        print(line)
        continue
    
    the_trip = line[0]
    
    if the_trip == '3':
        the_trip = '03'
    
    elif the_trip == '4':
        the_trip = '04'
        
    elif the_trip == '5':
        the_trip = '05'
        
    elif the_trip == '6':
        the_trip = '06'
    
    elif the_trip == '7':
        the_trip = '07'
    
    elif the_trip == '8':
        the_trip = '08'
    
    elif the_trip == '9':
        the_trip = '09'
    
    the_visit = line[1]
    the_weekday = line[2]
    the_upc = line[3]
    the_scan_count = int(line[4])
    the_dep = line[5]
    the_fline = line[6]
    
    train_visit_set.add(the_visit)
    
    trip_set.add(the_trip)
    visit_set.add(the_visit)
    weekday_set.add(the_weekday)
    upc_set.add(the_upc)
    dep_set.add(the_dep)
    fline_set.add(the_fline)
    
    visit_upc_scan_counter[the_visit + '@@' + the_upc] += the_scan_count
    visit_dep_scan_counter[the_visit + '@@' + the_dep] += the_scan_count
    visit_fline_scan_counter[the_visit + '@@' + the_fline] += the_scan_count
    
    visit2weekday[the_visit] = the_weekday
    visit2trip[the_visit] = the_trip
    
    
    if the_trip not in trip_dep_counters:
        trip_upc_counters[the_trip] = collections.Counter()
        trip_dep_counters[the_trip] = collections.Counter()
        trip_fline_counters[the_trip] = collections.Counter()
    
    
    trip_upc_counters[the_trip][the_upc] += 1
    trip_dep_counters[the_trip][the_dep] += 1
    trip_fline_counters[the_trip][the_fline] += 1
        
    
f.close()






critical_upcs = set()
critical_deps = set()
critical_flines = set()


critical_rank_thresh = 200

for trip in trip_upc_counters:
    for upc, num in trip_upc_counters[trip].most_common(critical_rank_thresh):
        critical_upcs.add(upc)


for trip in trip_dep_counters:
    for dep, num in trip_dep_counters[trip].most_common(critical_rank_thresh):
        critical_deps.add(dep)


for trip in trip_upc_counters:
    for fline, num in trip_fline_counters[trip].most_common(critical_rank_thresh):
        critical_flines.add(fline)






print('\n')
print('load test')
f = open(os.path.join(data_path, 'test.csv'))

counter = 0

for line in csv.reader(f):
    
    counter += 1
    
    if counter == 1:
        print(line)
        continue
    
    the_visit = line[0]
    the_weekday = line[1]
    the_upc = line[2]
    the_scan_count = int(line[3])
    the_dep = line[4]
    the_fline = line[5]
    
    test_visit_set.add(the_visit)
    
    visit_set.add(the_visit)
    weekday_set.add(the_weekday)
    upc_set.add(the_upc)
    dep_set.add(the_dep)
    fline_set.add(the_fline)
    
    visit_upc_scan_counter[the_visit + '@@' + the_upc] += the_scan_count
    visit_dep_scan_counter[the_visit + '@@' + the_dep] += the_scan_count
    visit_fline_scan_counter[the_visit + '@@' + the_fline] += the_scan_count
    
    visit2weekday[the_visit] = the_weekday
    
f.close()


train_visit_list = sorted(train_visit_set)
test_visit_list = sorted(test_visit_set)

trip_list = sorted(trip_set)
visit_list = sorted(visit_set)
weekday_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
upc_list = sorted(upc_set)
dep_list = sorted(dep_set)
fline_list = sorted(fline_set)



critical_upc_list = sorted(critical_upcs)
critical_dep_list = sorted(critical_deps)
critical_fline_list = sorted(critical_flines)


n_trips = len(trip_list)
n_visits = len(visit_list)
n_weekdays = len(weekday_list)
n_upcs = len(upc_list)
n_deps = len(dep_list)
n_flines = len(fline_list)

n_critical_upcs = len(critical_upc_list)
n_critical_deps = len(critical_dep_list)
n_critical_flines = len(critical_fline_list)


print('\n')
print('n_trips:', n_trips)
print('n_visits', n_visits)
print('n_weekdays:', n_weekdays)
print('n_upcs:', n_upcs)
print('n_deps:', n_deps)
print('n_flines:', n_flines)

print('n_critical_upcs', n_critical_upcs )
print('n_critical_deps', n_critical_deps)
print('n_critical_flines', n_critical_flines)

print('\n')


trip2trip_idx = {}
for i in range(n_trips):
    trip2trip_idx[trip_list[i]] = i

visit2visit_idx = {}
for i in range(n_visits):
    visit2visit_idx[visit_list[i]] = i

weekday2weekday_idx = {}
for i in range(n_weekdays):
    weekday2weekday_idx[weekday_list[i]] = i

upc2upc_idx = {}
for i in range(n_upcs):
    upc2upc_idx[upc_list[i]] = i

dep2dep_idx = {}
for i in range(n_deps):
    dep2dep_idx[dep_list[i]] = i

fline2fline_idx = {}
for i in range(n_flines):
    fline2fline_idx[fline_list[i]] = i



critical_upc2cupc_idx = {}
for i in range(n_critical_upcs):
    critical_upc2cupc_idx[critical_upc_list[i]] = i

critical_dep2cdep_idx = {}
for i in range(n_critical_deps):
    critical_dep2cdep_idx[critical_dep_list[i]] = i

critical_fline2cfline_idx = {}
for i in range(n_critical_flines):
    critical_fline2cfline_idx[critical_fline_list[i]] = i
    


dim_num = 10


rows = []
cols = []
values = []

for key in sorted(visit_upc_scan_counter):
    
    visit = key.split('@@')[0]
    visit = int(visit2visit_idx[visit])
    
    upc = key.split('@@')[1]
    
    
    try:
        upc = int(critical_upc2cupc_idx[upc])
    except:
        continue
    
    scan_counter = visit_upc_scan_counter[key]
    
    if scan_counter < 0:
        continue
    
    rows.append(visit)
    cols.append(upc)
    values.append(scan_counter)


visit_upc_sparse = csr_matrix((values, (rows, cols)), shape = (n_visits, n_critical_upcs), dtype = float)



print('\n')
print('NMF upc purchase')
nmf = NMF(n_components = dim_num)
visit_upc_decompose_purchase = nmf.fit_transform(visit_upc_sparse)




rows = []
cols = []
values = []

for key in sorted(visit_upc_scan_counter):
    
    visit = key.split('@@')[0]
    visit = int(visit2visit_idx[visit])
    
    upc = key.split('@@')[1]
    
    
    try:
        upc = int(critical_upc2cupc_idx[upc])
    except:
        continue
    
    scan_counter = visit_upc_scan_counter[key]
    
    if scan_counter >= 0:
        continue
    
    rows.append(visit)
    cols.append(upc)
    values.append(np.abs(scan_counter))


visit_upc_sparse = csr_matrix((values, (rows, cols)), shape = (n_visits, n_critical_upcs), dtype = float)



print('\n')
print('NMF upc return')
nmf = NMF(n_components = dim_num)
visit_upc_decompose_return = nmf.fit_transform(visit_upc_sparse)




rows = []
cols = []
values = []

for key in sorted(visit_dep_scan_counter):
    
    visit = key.split('@@')[0]
    visit = int(visit2visit_idx[visit])
    
    dep = key.split('@@')[1]
    
    try:
        dep = int(critical_dep2cdep_idx[dep])
    except:
        continue
    
    scan_counter = visit_dep_scan_counter[key]
    
    if scan_counter < 0:
        continue
    
    rows.append(visit)
    cols.append(dep)
    values.append(scan_counter)


visit_dep_sparse = csr_matrix((values, (rows, cols)), shape = (n_visits, n_critical_deps), dtype = float)



print('\n')
print('NMF dep purchase')
nmf = NMF(n_components = dim_num)
visit_dep_decompose_purchase = nmf.fit_transform(visit_dep_sparse)




rows = []
cols = []
values = []

for key in sorted(visit_fline_scan_counter):
    
    visit = key.split('@@')[0]
    visit = int(visit2visit_idx[visit])
    
    fline = key.split('@@')[1]
    
    try:
        fline = int(critical_fline2cfline_idx[fline])
    except:
        continue
    
    scan_counter = visit_fline_scan_counter[key]
    
    if scan_counter < 0:
        continue
    
    rows.append(visit)
    cols.append(fline)
    values.append(scan_counter)


visit_fline_sparse = csr_matrix((values, (rows, cols)), shape = (n_visits, n_critical_flines), dtype = float)



print('\n')
print('NMF fline purchase')
nmf = NMF(n_components = dim_num)
visit_fline_decompose_purchase = nmf.fit_transform(visit_fline_sparse)




rows = []
cols = []
values = []

for key in sorted(visit_fline_scan_counter):
    
    visit = key.split('@@')[0]
    visit = int(visit2visit_idx[visit])
    
    fline = key.split('@@')[1]
    
    try:
        fline = int(critical_fline2cfline_idx[fline])
    except:
        continue
    
    scan_counter = visit_fline_scan_counter[key]
    
    if scan_counter >= 0:
        continue
    
    rows.append(visit)
    cols.append(fline)
    values.append(np.abs(scan_counter))


visit_fline_sparse = csr_matrix((values, (rows, cols)), shape=(n_visits, n_critical_flines), dtype = float)



print('\n')
print('NMF fline return')
nmf = NMF(n_components = dim_num)
visit_fline_decompose_return = nmf.fit_transform(visit_fline_sparse)



xs = []
ys = []


for visit in train_visit_list:
    
    x = []
    
    x.append(int(visit))
    
    weekday = visit2weekday[visit]
    weekday_vec = np.zeros(n_weekdays)
    weekday_vec[weekday2weekday_idx[weekday]] = 1
    
    x = x + list(weekday_vec)
    x = x + list(visit_upc_decompose_purchase[visit2visit_idx[visit],:])
    x = x + list(visit_upc_decompose_return[visit2visit_idx[visit],:])
    x = x + list(visit_dep_decompose_purchase[visit2visit_idx[visit],:])
    x = x + list(visit_fline_decompose_purchase[visit2visit_idx[visit],:])
    x = x + list(visit_fline_decompose_return[visit2visit_idx[visit],:])
    
    xs.append(x)
    
    
    trip = visit2trip[visit]
#    trip_vec = np.zeros(n_trips)
#    trip_vec[trip2trip_idx[trip]] = 1
    
    y = 'the_' + trip
    
    ys.append(y)


xs = np.asarray(xs)
ys = np.asarray(ys)


xs_test = []


for visit in test_visit_list:
    
    x = []
    
    x.append(int(visit))
    
    weekday = visit2weekday[visit]
    weekday_vec = np.zeros(n_weekdays)
    weekday_vec[weekday2weekday_idx[weekday]] = 1
    
    x = x + list(weekday_vec)
    x = x + list(visit_upc_decompose_purchase[visit2visit_idx[visit],:])
    x = x + list(visit_upc_decompose_return[visit2visit_idx[visit],:])
    x = x + list(visit_dep_decompose_purchase[visit2visit_idx[visit],:])
    x = x + list(visit_fline_decompose_purchase[visit2visit_idx[visit],:])
    x = x + list(visit_fline_decompose_return[visit2visit_idx[visit],:])
    
    xs_test.append(x)


xs_test = np.asarray(xs_test)

print('\n')
print('xs shape', xs.shape)
print('ys shape', ys.shape)



xs_mean = np.mean(np.vstack([xs, xs_test]), axis = 0)
xs_std = np.std(np.vstack([xs, xs_test]), axis = 0)

xs -= xs_mean
xs /= xs_std

xs_test -= xs_mean
xs_test /= xs_std




permu_idx = np.random.permutation(len(xs))
xs = xs[permu_idx,:]
ys = ys[permu_idx]


#sys.exit()



"""
#################################################
# modle building
#################################################

n_estimators_num  =  5

clf0 = xgb.XGBClassifier(n_estimators = n_estimators_num, n_jobs = -1)
clf1 = lgb.LGBMClassifier(n_estimators = n_estimators_num, n_jobs = -1, verbose = 0)

clf2 = RandomForestClassifier(n_estimators = n_estimators_num, n_jobs = -1, verbose = 0)
clf3 = ExtraTreesClassifier(n_estimators = n_estimators_num, n_jobs = -1, verbose = 0)
clf4 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators = n_estimators_num)

clf5 = DecisionTreeClassifier()

#clf6 = svm.SVC(kernel = 'linear', max_iter = 200, gamma = 'auto', verbose = 1)

#clf7 = KNeighborsClassifier(n_neighbors = 3)

clf8 = MLPClassifier(early_stopping = True, tol = 0.0001, verbose = 1)

clfs = [clf5]



#################################################
# scoring metrics
#################################################


scoring = [
            'accuracy', 
            'f1_macro', 'f1_micro', 
            'precision_macro','precision_micro', 
            'recall_macro', 'recall_micro'
           ]



#################################################
# scoring metrics
#################################################




xs = xs[:,:]
ys = ys[:]

for clf in clfs:
    
    print('\n')
    print('Time Taken:', time.time() - stime)
    print('---------------------------------------------------')
    print(clf)
    
    scores = cross_validate(clf,
                            xs, ys,
                            scoring = scoring,
#                            cv = 3,
                            cv = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 1, random_state = 46),
                            return_train_score = True,
                            n_jobs = -1,
                            verbose = 1
                            )
    
    the_score_keys = sorted(scores.keys())
    
    for the_score_key in the_score_keys:
        the_scores = scores[the_score_key]
        print(the_score_key, ": %0.2f (+/- %0.2f)" % (the_scores.mean(), the_scores.std() * 2))


"""


rskf = RepeatedStratifiedKFold(n_splits = 4, n_repeats = 3, random_state = 46)


model2scores = {}
model2test_preds = {}

for train_index, valid_index in rskf.split(xs, ys):
    
    xs_train, xs_valid = xs[train_index,:], xs[valid_index,:]
    ys_train, ys_valid = ys[train_index], ys[valid_index]
    
    #################################################
    # modle building
    #################################################
    
    n_estimators_num  =  500
    
    clf0 = xgb.XGBClassifier(n_estimators = n_estimators_num, objective = 'multi:softprob', n_jobs = -1)
    clf1 = lgb.LGBMClassifier(n_estimators = n_estimators_num, objective = 'multiclass', n_jobs = -1, verbose = 0)
    
    clf2 = RandomForestClassifier(n_estimators = n_estimators_num, n_jobs = -1, verbose = 0)
    clf3 = ExtraTreesClassifier(n_estimators = n_estimators_num, n_jobs = -1, verbose = 0)
    clf4 = AdaBoostClassifier(n_estimators = n_estimators_num)
    
    clf5 = DecisionTreeClassifier()
    
    clf6 = svm.SVC(kernel = 'linear', max_iter = 200, gamma = 'auto', probability = True, verbose = 1)
    
    #clf7 = KNeighborsClassifier(n_neighbors = 3)
    
    clf8 = MLPClassifier(hidden_layer_sizes = (256, 256, 128, 128), early_stopping = True, tol = 0.0001, verbose = 0)
    
#    clfs = [clf5]
#    clf_names = ['DT']
    
#    clfs = [clf3]
#    clf_names = ['EXT']
    
#    clfs = [clf6]
#    clf_names = ['SVM']
    
    clfs = [clf1, clf2, clf3, clf4, clf5, clf8]
    clf_names = ['LGB', 'RF', 'EXT', 'ADA', 'DT', 'MLP']
    
    
    for clf, clf_name in zip(clfs, clf_names):
        
        
        print('------------------------------------------')
        print(clf)
        print('\n')
        
        if clf_name not in model2scores:
            
            model2scores[clf_name] = {}
            
            model2scores[clf_name]['fit_time'] = []
            model2scores[clf_name]['score_time'] = []
            
            model2scores[clf_name]['train_accuracy'] = []
            model2scores[clf_name]['train_logloss'] = []
            model2scores[clf_name]['train_f1_macro'] = []
            model2scores[clf_name]['train_f1_micro'] = []
            model2scores[clf_name]['train_precision_macro'] = []
            model2scores[clf_name]['train_precision_micro'] = []
            model2scores[clf_name]['train_recall_macro'] = []
            model2scores[clf_name]['train_recall_micro'] = []
            
            model2scores[clf_name]['valid_accuracy'] = []
            model2scores[clf_name]['valid_logloss'] = []
            model2scores[clf_name]['valid_f1_macro'] = []
            model2scores[clf_name]['valid_f1_micro'] = []
            model2scores[clf_name]['valid_precision_macro'] = []
            model2scores[clf_name]['valid_precision_micro'] = []
            model2scores[clf_name]['valid_recall_macro'] = []
            model2scores[clf_name]['valid_recall_micro'] = []
            
            model2test_preds[clf_name] = []
        
        
        
        # model fit
        
        clf_fit_stime = time.time()
        
        clf.fit(xs_train, ys_train)
        
        fit_time = time.time() - clf_fit_stime
        
        
        # train prediction
        
        y_train_pred = clf.predict(xs_train)
        y_train_pred_proba = clf.predict_proba(xs_train)
        
        y_train_true = ys_train
        
        acc_train = accuracy_score(y_train_true, y_train_pred)
        logloss_train = log_loss(y_train_true, y_train_pred_proba)
        
        f1_macro_train = f1_score(y_train_true, y_train_pred, average = 'macro')
        f1_micro_train = f1_score(y_train_true, y_train_pred, average = 'micro')
        
        precision_macro_train = precision_score(y_train_true, y_train_pred, average = 'macro')
        precision_micro_train = precision_score(y_train_true, y_train_pred, average = 'micro')
        
        recall_macro_train = recall_score(y_train_true, y_train_pred, average = 'macro')
        recall_micro_train = recall_score(y_train_true, y_train_pred, average = 'micro')
        
        
        
        # valid prediction
        
        clf_score_stime = time.time()
        
        y_valid_pred = clf.predict(xs_valid)
        
        score_time = time.time() - clf_score_stime
        
        y_valid_pred_proba = clf.predict_proba(xs_valid)
        
        y_valid_true = ys_valid
        
        acc_valid = accuracy_score(y_valid_true, y_valid_pred)
        logloss_valid = log_loss(y_valid_true, y_valid_pred_proba)
        
        f1_macro_valid = f1_score(y_valid_true, y_valid_pred, average = 'macro')
        f1_micro_valid = f1_score(y_valid_true, y_valid_pred, average = 'micro')
        
        precision_macro_valid = precision_score(y_valid_true, y_valid_pred, average = 'macro')
        precision_micro_valid = precision_score(y_valid_true, y_valid_pred, average = 'micro')
        
        recall_macro_valid = recall_score(y_valid_true, y_valid_pred, average = 'macro')
        recall_micro_valid = recall_score(y_valid_true, y_valid_pred, average = 'micro')
        
        
        
        # test prediction
        y_test_pred_proba = clf.predict_proba(xs_test)
        
        
        
        
        print('Time')
        print('Fit:', fit_time)
        print('Score:', score_time)
        
        
        print('Accuracy')
        print('Train:', acc_train)
        print('Valid:', acc_valid)
        print('\n')
        
        print('Log Loss')
        print('Train:', logloss_train)
        print('Valid:', logloss_valid)
        print('\n')
        
        print('F1 Macro')
        print('Train:', f1_macro_train)
        print('Valid:', f1_macro_valid)
        print('\n')
        
        print('F1 Micro')
        print('Train:', f1_micro_train)
        print('Valid:', f1_micro_valid)
        print('\n')
        
        print('Precision Macro')
        print('Train:', precision_macro_train)
        print('Valid:', precision_macro_valid)
        print('\n')
        
        print('Precision Micro')
        print('Train:', precision_micro_train)
        print('Valid:', precision_micro_valid)
        print('\n')
        
        print('Recall Macro')
        print('Train:', recall_macro_train)
        print('Valid:', recall_macro_valid)
        print('\n')
        
        print('Recall Micro')
        print('Train:', recall_micro_train)
        print('Valid:', recall_micro_valid)
        print('\n')
        
        
        model2scores[clf_name]['fit_time'].append(fit_time)
        model2scores[clf_name]['score_time'].append(score_time)
        
        model2scores[clf_name]['train_accuracy'].append(acc_train)
        model2scores[clf_name]['train_logloss'].append(logloss_train)
        model2scores[clf_name]['train_f1_macro'].append(f1_macro_train)
        model2scores[clf_name]['train_f1_micro'].append(f1_micro_train)
        model2scores[clf_name]['train_precision_macro'].append(precision_macro_train)
        model2scores[clf_name]['train_precision_micro'].append(precision_micro_train)
        model2scores[clf_name]['train_recall_macro'].append(recall_macro_train)
        model2scores[clf_name]['train_recall_micro'].append(recall_micro_train)
        
        model2scores[clf_name]['valid_accuracy'].append(acc_valid)
        model2scores[clf_name]['valid_logloss'].append(logloss_valid)
        model2scores[clf_name]['valid_f1_macro'].append(f1_macro_valid)
        model2scores[clf_name]['valid_f1_micro'].append(f1_micro_valid)
        model2scores[clf_name]['valid_precision_macro'].append(precision_macro_valid)
        model2scores[clf_name]['valid_precision_micro'].append(precision_micro_valid)
        model2scores[clf_name]['valid_recall_macro'].append(recall_macro_valid)
        model2scores[clf_name]['valid_recall_micro'].append(recall_micro_valid)
        
        model2test_preds[clf_name].append(y_test_pred_proba)
        
        
        

for model in model2scores:
    
    print('\n')
    print(model)
    
    scores = model2scores[model]
    
    the_score_keys = sorted(scores.keys())
    
    for the_score_key in the_score_keys:
        the_scores = np.asarray(scores[the_score_key])
        print(the_score_key, ": %0.2f (+/- %0.2f)" % (the_scores.mean(), the_scores.std() * 2))


with open('model2scores.pickle', 'wb') as fp:
    pickle.dump(model2scores, fp)

with open('model2scores.pickle', 'rb') as fp:
    model2scores = pickle.load(fp)
    
    


model2test_pred_mean = {}

for model in model2test_preds:
    
    the_test_preds = model2test_preds[model]
    the_test_pred_mean = np.zeros((len(xs_test), n_trips))
    for the_test_pred in the_test_preds:
        the_test_pred_mean = the_test_pred_mean + np.asarray(the_test_pred)
    the_test_pred_mean /= len(the_test_preds)
    
    model2test_pred_mean[model] = the_test_pred_mean
        

the_result = model2test_pred_mean['LGB']
the_result /= 1

f = open('submit.csv', 'w')

f.write('"VisitNumber","TripType_3","TripType_4","TripType_5","TripType_6","TripType_7","TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18","TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24","TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30","TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36","TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42","TripType_43","TripType_44","TripType_999"\n')
for i in range(len(test_visit_list)):
    f.write(str(int(test_visit_list[i])))
    for j in the_result[i]:
        f.write(',')
        f.write(str(j))
    f.write('\n')
f.close()

print('Time Taken:', time.time() - stime)
