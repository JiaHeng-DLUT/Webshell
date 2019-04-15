# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:51:17 2019

@author: jiaheng
"""
import os

import numpy as np
import pandas as pd
# from svmutil import *
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 项目所在路径
projectPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


file = "data.txt"
# load data
data = pd.read_csv(projectPath + "/res/" + file, header=None)
data = data.to_numpy()
X = data[:, :-1]
y = data[:, -1]

# split data set into 3 parts: train, cr and test
Xtrain, Xt, ytrain, yt = train_test_split(X, y, test_size=0.4, random_state=0)
Xcr, Xtest, ycr, ytest = train_test_split(Xt, yt, test_size=0.5, random_state=0)

# train
maxF = 0
model = 0
with open(projectPath + "/res/" + "result" + file, mode="w") as f:
    f.write("")
with open(projectPath + "/res/" + "result" + file, mode="a") as f:
    for c in np.arange(9, 9.01, 0.05):
        for g in np.arange(0.04, 0.05, 0.005):
            print("c: %f  g: %f" % (c, g))
            f.write("c: %f  g: %f\n" % (c, g))
            clf = SVC(C=c, gamma=g)
            clf.fit(Xtrain, ytrain)
            predictedLabelCR = clf.predict(Xcr)
            report = metrics.classification_report(ycr, predictedLabelCR, output_dict=True)
            # print(report.keys())
            F = report['1.0']['f1-score']
            if F > maxF:
                maxF = F
                model = clf
                print(report['1.0'])
                f.write(metrics.classification_report(ycr, predictedLabelCR))
                A = metrics.accuracy_score(ycr, predictedLabelCR)
                f.write("A:      %f" % A)
    # output
    print("The best parameters are as follows:")
    print("C:      %f" % (model.C))
    print("gamma:  %f" % (model.gamma))
    f.write("The best parameters are as follows:\n")
    f.write("C:      %f\n" % (model.C))
    f.write("gamma:  %f\n" % (model.gamma))
    predictedLabelTest = clf.predict(Xtest)
    for i in range(0,50):
        print(str(predictedLabelTest[i])+"    "+str(ytest[i]))
    reportTest = metrics.classification_report(ytest, predictedLabelTest, output_dict=True)
    A = metrics.accuracy_score(ytest, predictedLabelTest)
    print(reportTest)
    print("A:      %f" % A)
    f.write(metrics.classification_report(ytest, predictedLabelTest))
    f.write("A:      %f" % A)
    joblib.dump(model, projectPath + "/res/model.joblib")
    