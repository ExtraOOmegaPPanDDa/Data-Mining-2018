# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:35:32 2018

@author: user
"""

import pickle
import ggplot
import pandas as pd
import numpy as np


with open('model2scores.pickle', 'rb') as fp:
    model2scores = pickle.load(fp)
    


for plot_key_name in model2scores['LGB'].keys():
    
    plot_dataset = []
    
    for model in model2scores:
        
        print('\n')
        print(model)
        
        the_mean = np.mean(model2scores[model][plot_key_name])
        the_std = np.std(model2scores[model][plot_key_name])
        the_max = np.max(model2scores[model][plot_key_name])
        the_min = np.min(model2scores[model][plot_key_name])
        
        
        total = len(model2scores[model][plot_key_name])
        for value in model2scores[model][plot_key_name]:
            plot_dataset.append([model, value, the_mean/total, the_std, the_max, the_min])
    
    
    plot_dataset_pd = pd.DataFrame(plot_dataset, columns = ['model', 'value', 'weight', 'std', 'max', 'min'])
    
    
    if 'logloss' in plot_key_name:
        
        p = ggplot.ggplot(ggplot.aes(x = 'model', fill = 'model', weight = 'weight'), data = plot_dataset_pd) +\
        ggplot.geom_bar(position = 'stack', width = 4) +\
        ggplot.geom_errorbar(ggplot.aes(x = 'model', y = 'value')) +\
        ggplot.ylim(0 ,5.05) +\
        ggplot.ggtitle(plot_key_name)
        
        #print(p)
        
    elif 'time' in plot_key_name:
        
        p = ggplot.ggplot(ggplot.aes(x = 'model', fill = 'model', weight = 'weight'), data = plot_dataset_pd) +\
        ggplot.geom_bar(position = 'stack', width = 4) +\
        ggplot.geom_errorbar(ggplot.aes(x = 'model', y = 'value')) +\
        ggplot.ggtitle(plot_key_name)
        
        #print(p)
    
    else:
        
        p = ggplot.ggplot(ggplot.aes(x = 'model', fill = 'model', weight = 'weight'), data = plot_dataset_pd) +\
        ggplot.geom_bar(position = 'stack', width = 4) +\
        ggplot.geom_errorbar(ggplot.aes(x = 'model', y = 'value')) +\
        ggplot.ylim(0 ,1.05) +\
        ggplot.ggtitle(plot_key_name)
        
        #print(p)
        
    
    
    p.save(plot_key_name + '.png')