# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 03:56:52 2018

@author: mei
"""


import pandas as pd 
#import lib.preproses as pp
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import cross_validate#, GridSearchCV
import warnings
warnings.filterwarnings("ignore")

### load label dan juga fitur
print("Load label dan fitur")
label = pickle.load(open("label.pkl", "rb"))
fasttext_feat = pickle.load(open("fitur_ft.pkl","rb"))
y = np.array(label)


print("proses training dan testing")
##------ pake max entropy -----
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
lr = LogisticRegression(multi_class='ovr')
#lr2 = GridSearchCV(lr, param_grid)

scoring = ['accuracy','f1_macro']
results_lr = cross_validate(lr, fasttext_feat, y, cv=5, scoring=scoring)
print("------- Hasil Evaluasi ------------")
print("Accuracy = ", np.mean(results_lr['test_accuracy']))
print("F1 Score = ", np.mean(results_lr['test_f1_macro']))


