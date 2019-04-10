# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:51:17 2019

@author: jiaheng
"""
import os

import numpy as np
import pandas as pd
# from svmutil import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# 项目所在路径
projectPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# load data
data = pd.read_csv(projectPath + "/res/data.txt", header=None)
data = data.to_numpy()
X = data[:, 0:-2]
y = data[:, -1]
# split data set into 3 parts: train, cr and test
Xtrain, Xt, ytrain, yt = train_test_split(X, y, test_size=0.4, random_state=0)
Xcr, Xtest, ycr, ytest = train_test_split(Xt, yt, test_size=0.5, random_state=0)
# train
maxF = 0
maxP = 0
maxR = 0
maxA = 0
model = 0
for c in np.arange(11.65, 11.75, 0.01):
    for g in np.arange(0.1, 0.21, 0.02):
        print("c: %f  g: %f" % (c, g))
        clf = SVC(C=c, gamma=g)
        clf.fit(Xtrain, ytrain)
        predictedLabel = clf.predict(Xcr)
        P = sum(predictedLabel*ycr)/sum(predictedLabel)
        R = sum(predictedLabel*ycr)/sum(ycr)
        F = 2*P*R/(P+R)
        if F > maxF:
            maxF = F
            maxP = P
            maxR = R
            maxA = sum(predictedLabel==ycr)/ycr.size
            model = clf
            # output
            print("F:  %f" % maxF)
            print("P:  %f" % maxP)
            print("R:  %f" % maxR)
            print("A:  %f" % maxA)
# output
print("The best parameters are as follows:")
print("C:      %f" % (model.C))
print("gamma:  %f" % (model.gamma)) 
predictedLabel = clf.predict(Xtest)
P = sum(predictedLabel*ytest)/sum(predictedLabel)
R = sum(predictedLabel*ytest)/sum(ytest)
F = 2*P*R/(P+R)
A = sum(predictedLabel==ycr)/ytest.size
# output
print("F:    %f" % maxF)
print("P:    %f" % maxP)
print("R:    %f" % maxR)
print("A:    %f" % maxA)        
