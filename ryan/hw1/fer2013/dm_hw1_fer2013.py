#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:00:28 2018

@author: HSIN
"""

import sys
import csv
import time
import numpy as np
import openpyxl
import matplotlib.pyplot as plt


from sklearn.utils import shuffle

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics


stime = time.time()

np.random.seed(46)
eps = 1e-05

load_path = './fer2013.csv'
#(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

x = []
y = []

f = open(load_path)
for line in csv.reader(f):
    if line[0] in ['3','4']:
        x.append([float(a) for a in line[1].split(' ')])
        y.append(line[0])


x = np.asarray(x)
y = np.asarray(y)

print('x shape', x.shape)
print('y shape', y.shape)

print('Category', set(y))


# suffle data
x, y = shuffle(x, y)



plt.imshow(x[0].reshape(48,48), cmap = 'gray')


# PCA Components Plot

nrow, ncol = 8, 8
pca = PCA(n_components = nrow * ncol)
x_pca = pca.fit_transform(x)

w = 10
h = 10
fig = plt.figure(figsize=(8, 8))

for i in range(nrow * ncol):
    fig.add_subplot(nrow, ncol, i + 1)
    plt.imshow(pca.components_[i,:].reshape(48,48), cmap = 'gray')
#plt.show()
plt.savefig('pca_face_components.png')


# NMF Components Plot

nrow, ncol = 8, 8
nmf = NMF(n_components = nrow * ncol)
x_nmf = nmf.fit_transform(x)

w = 10
h = 10
fig = plt.figure(figsize=(8, 8))

for i in range(nrow * ncol):
    fig.add_subplot(nrow, ncol, i + 1)
    plt.imshow(nmf.components_[i,:].reshape(48,48), cmap = 'gray')
#plt.show()
plt.savefig('nmf_face_components.png')


#sys.exit()

# modle building
svm = svm.LinearSVC()
rf = RandomForestClassifier(n_jobs = -1)
ext = ExtraTreesClassifier(n_jobs = -1)
#xgb = XGBClassifier(n_jobs = -1)



clfs = [svm, rf, ext]
dims = [x.shape[1],128,64,32,16,8,4,2]

#scoring = ['accuracy','f1','precision','recall','roc_auc']
scoring = ['accuracy','f1_micro','f1_macro']


for dim in dims:  
    
    
    if dim == x.shape[1]:
        
        
        
        #################################################
        ################ NO Decomposition ###############
        #################################################
        
        print('-----------------------------------------------------')
        print('No Decomposition')
        
        print('x shape', x.shape)
        print('y shape', y.shape)
        
        for clf in clfs:
            
            print('-----------------------------------------------------')
            print('The Classifier')
            print(clf)
            print('-----------------------------------------------------')
            
            
        
            scores = cross_validate(clf,
                                    x, y,
                                    scoring = scoring,
                                    cv = 10,
                                    return_train_score = True,
                                    n_jobs = -1,
                                    verbose = 1
                                    )
            
            the_score_keys = sorted(scores.keys())
            
            for the_score_key in the_score_keys:
                the_scores = scores[the_score_key]
                print(the_score_key, ": %0.2f (+/- %0.2f)" % (the_scores.mean(), the_scores.std() * 2))
            
            print('-----------------------------------------------------')
    
    else:
        
            #################################################
            ################ PCA Decomposition ##############
            #################################################
        
            print('-----------------------------------------------------')
            print('PCA Decomposition', dim)
            
            pca = PCA(n_components = dim)
            x_pca = pca.fit_transform(x)
            y_pca = y
            
            
            print('x_pca shape', x_pca.shape)
            print('y_pca shape', y_pca.shape)
            
            for clf in clfs:
            
                print('-----------------------------------------------------')
                print('The Classifier')
                print(clf)
                print('-----------------------------------------------------')
                
                
            
                scores = cross_validate(clf,
                                        x_pca, y_pca,
                                        scoring = scoring,
                                        cv = 10,
                                        return_train_score = True,
                                        n_jobs = -1,
                                        verbose = 1
                                        )
                
                the_score_keys = sorted(scores.keys())
                
                for the_score_key in the_score_keys:
                    the_scores = scores[the_score_key]
                    print(the_score_key, ": %0.2f (+/- %0.2f)" % (the_scores.mean(), the_scores.std() * 2))
                
                print('-----------------------------------------------------')
            
            
            
            
            #################################################
            ################ NMF Decomposition ##############
            #################################################
            
            
            print('-----------------------------------------------------')
            print('NMF Decomposition', dim)
            
            nmf = NMF(n_components = dim)
            x_nmf = nmf.fit_transform(x)
            y_nmf = y
            
            
            print('x_nmf shape', x_nmf.shape)
            print('y_nmf shape', y_nmf.shape)
        
            
            
            
            for clf in clfs:
            
                print('-----------------------------------------------------')
                print('The Classifier')
                print(clf)
                print('-----------------------------------------------------')
                
                
                scores = cross_validate(clf,
                                            x_pca, y_pca,
                                            scoring = scoring,
                                            cv = 10,
                                            return_train_score = True,
                                            n_jobs = -1,
                                            verbose = 1
                                        )
                
                the_score_keys = sorted(scores.keys())
                
                for the_score_key in the_score_keys:
                    the_scores = scores[the_score_key]
                    print(the_score_key, ": %0.2f (+/- %0.2f)" % (the_scores.mean(), the_scores.std() * 2))
                
                print('-----------------------------------------------------')
    
    
    
    
    print('\n\n')




print('All Time Taken:', time.time()-stime)