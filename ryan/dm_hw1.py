#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:00:28 2018

@author: HSIN
"""

import sys
import time
import numpy as np
import openpyxl
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

stime = time.time()

np.random.seed(46)
eps = 1e-05
n_estimator = 100

load_path = './super_market_data.xlsx'


wb = openpyxl.load_workbook(load_path)

ws = wb['交易記錄檔']

ws_max_row = ws.max_row
ws_max_col = ws.max_column


mids = set()
items = set()
item_classes = set()

for i in range(ws_max_row):
    
    the_row_idx = i + 1
    
    if the_row_idx == 1:
        continue
    
    mids.add(ws['B' + str(the_row_idx)].value)
    items.add(ws['F' + str(the_row_idx)].value)
    item_classes.add(ws['E' + str(the_row_idx)].value)


mids = sorted(list(mids))
items = sorted(list(items))
item_classes = sorted(list(item_classes))

x = np.zeros((len(mids),len(items)))
y = np.zeros(len(mids))

print('x shape', x.shape)
print('y shape', y.shape)


for i in range(ws_max_row):
    
    the_row_idx = i + 1
    
    if the_row_idx == 1:
        continue
    
    mid = ws['B' + str(the_row_idx)].value
    item = ws['F' + str(the_row_idx)].value
    item_class = ws['E' + str(the_row_idx)].value
    
    num = int(ws['G' + str(the_row_idx)].value)
    
    
    x[mids.index(mid), items.index(item)] += num
    
    

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if x[i,j] < 0:
#            print(mids[i], items[j], x[i,j])
#            sys.exit()
            x[i,j] = 0




ws = wb['會員資料檔']

ws_max_row = ws.max_row
ws_max_col = ws.max_column


for i in range(ws_max_row):
    
    the_row_idx = i + 1
    
    if the_row_idx == 1:
        continue
    
    mid = ws['A' + str(the_row_idx)].value
    sex = ws['C' + str(the_row_idx)].value
    marry = ws['G' + str(the_row_idx)].value
    
    if sex == '男':
        y[mids.index(mid)] = 1
    elif sex == '女':
        y[mids.index(mid)] = 0
    else:
        print('Warning')
    
    
#    if marry == '已婚':
#        y[mids.index(mid)] = 1
#    else:
#        y[mids.index(mid)] = 0
        


print('----------------------------------------------------')
print('No Decomposition')

train_valid_ratio = 0.9
indices = np.random.permutation(x.shape[0])
train_idx, valid_idx = indices[:int(x.shape[0] * train_valid_ratio)], indices[int(x.shape[0] * train_valid_ratio):]
x_train, x_valid = x[train_idx,:], x[valid_idx,:]
y_train, y_valid = y[train_idx], y[valid_idx]


x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train) + eps

x_train = x_train - np.tile(x_train_mean,(len(x_train),1))
x_train = x_train/np.tile(x_train_std,(len(x_train),1))


x_valid = x_valid - np.tile(x_train_mean,(len(x_valid),1))
x_valid = x_valid/np.tile(x_train_std,(len(x_valid),1))

    
    
print('x train shape', x_train.shape)
print('y train shape', y_train.shape)

print('x valid shape', x_valid.shape)
print('y valid shape', y_valid.shape)


print('Building Model...')
    
clf = XGBClassifier()


print('Train Model...')
print('Time Taken:', time.time()-stime)

clf.fit(x_train, y_train)

print('Train Done')
print('Time Taken:', time.time()-stime)


train_pred = clf.predict_proba(x_train)[:,1]

train_acc = accuracy_score(y_train, np.round(train_pred))
print('Train ACC:', train_acc)
fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
train_auc = metrics.auc(fpr, tpr)
print('Train AUC:', train_auc)



valid_pred = clf.predict_proba(x_valid)[:,1]

valid_acc = accuracy_score(y_valid, np.round(valid_pred))
print('Valid ACC:', valid_acc)
fpr, tpr, thresholds = metrics.roc_curve(y_valid, valid_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('Valid AUC:', valid_auc)




for nmf_dim in [50, 30, 20, 15, 10, 5, 2]:
    
    print('----------------------------------------------------')
    print('NMF Decomposition', nmf_dim)
    
    
    nmf = NMF(n_components = nmf_dim)
    x_nmf = nmf.fit_transform(x)
    
    
    print('NMF Decomposition Done')
    
    train_valid_ratio = 0.9
    indices = np.random.permutation(x_nmf.shape[0])
    train_idx, valid_idx = indices[:int(x_nmf.shape[0] * train_valid_ratio)], indices[int(x_nmf.shape[0] * train_valid_ratio):]
    x_train, x_valid = x_nmf[train_idx,:], x_nmf[valid_idx,:]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    
    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train) + eps
    
    x_train = x_train - np.tile(x_train_mean,(len(x_train),1))
    x_train = x_train/np.tile(x_train_std,(len(x_train),1))
    
    
    x_valid = x_valid - np.tile(x_train_mean,(len(x_valid),1))
    x_valid = x_valid/np.tile(x_train_std,(len(x_valid),1))
    
    
    print('x train shape', x_train.shape)
    print('y train shape', y_train.shape)
    
    print('x valid shape', x_valid.shape)
    print('y valid shape', y_valid.shape)
    
    
    print('Building Model...')
        
    clf = XGBClassifier()
    
    
    print('Train Model...')
    print('Time Taken:', time.time()-stime)
    
    clf.fit(x_train, y_train)
    
    print('Train Done')
    print('Time Taken:', time.time()-stime)
    
    
    train_pred = clf.predict_proba(x_train)[:,1]
    
    train_acc = accuracy_score(y_train, np.round(train_pred))
    print('Train ACC:', train_acc)
    fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
    train_auc = metrics.auc(fpr, tpr)
    print('Train AUC:', train_auc)
    
    
    
    valid_pred = clf.predict_proba(x_valid)[:,1]
    
    valid_acc = accuracy_score(y_valid, np.round(valid_pred))
    print('Valid ACC:', valid_acc)
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, valid_pred, pos_label=1)
    valid_auc = metrics.auc(fpr, tpr)
    print('Valid AUC:', valid_auc)



for pca_dim in [500, 200, 100, 50, 30, 15, 8, 2]:
    
    print('----------------------------------------------------')
    print('PCA Decomposition', pca_dim)
    
    
    pca = PCA(n_components = pca_dim)
    pca.fit(x)
    
    x_pca = pca.transform(x)
    
    
    print('PCA Decomposition Done')
    
    train_valid_ratio = 0.9
    indices = np.random.permutation(x_pca.shape[0])
    train_idx, valid_idx = indices[:int(x_pca.shape[0] * train_valid_ratio)], indices[int(x_pca.shape[0] * train_valid_ratio):]
    x_train, x_valid = x_pca[train_idx,:], x_pca[valid_idx,:]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    
    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train) + eps
    
    x_train = x_train - np.tile(x_train_mean,(len(x_train),1))
    x_train = x_train/np.tile(x_train_std,(len(x_train),1))
    
    
    x_valid = x_valid - np.tile(x_train_mean,(len(x_valid),1))
    x_valid = x_valid/np.tile(x_train_std,(len(x_valid),1))
    
    
    print('x train shape', x_train.shape)
    print('y train shape', y_train.shape)
    
    print('x valid shape', x_valid.shape)
    print('y valid shape', y_valid.shape)
    
    
    print('Building Model...')
        
    clf = XGBClassifier()
    
    
    print('Train Model...')
    print('Time Taken:', time.time()-stime)
    
    clf.fit(x_train, y_train)
    
    print('Train Done')
    print('Time Taken:', time.time()-stime)
    
    
    train_pred = clf.predict_proba(x_train)[:,1]
    
    train_acc = accuracy_score(y_train, np.round(train_pred))
    print('Train ACC:', train_acc)
    fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
    train_auc = metrics.auc(fpr, tpr)
    print('Train AUC:', train_auc)
    
    
    
    valid_pred = clf.predict_proba(x_valid)[:,1]
    
    valid_acc = accuracy_score(y_valid, np.round(valid_pred))
    print('Valid ACC:', valid_acc)
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, valid_pred, pos_label=1)
    valid_auc = metrics.auc(fpr, tpr)
    print('Valid AUC:', valid_auc)