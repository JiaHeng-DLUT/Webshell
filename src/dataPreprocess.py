# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:51:17 2019

@author: jiaheng
"""
import os

import numpy as np
import pandas as pd

# 项目所在路径
projectPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


with open(projectPath + "/res/feature.csv") as file:
    feature = pd.read_csv(file)
    [m, n] = feature.shape
#        print(n)
    p1 = feature.iloc[:, :-1].median(axis=0)
    p2 = feature.iloc[:, :-1].mean(axis=0)
    print(p1)
    print(p2)
    p1.to_csv(projectPath + "/res/median.csv",index=False,header=False)
    p2.to_csv(projectPath + "/res/mean.csv",index=False,header=False)
    data = feature.copy()       
    for i in range(0, m):
        print("Example: " + str(i))
        for j in range(0, n-1):
            if feature.iloc[i, j] <= p1[j]:
                data.iloc[i, j] = 1
            else:
                if feature.iloc[i, j] <= p2[j]:
                    data.iloc[i, j] = 2
                else:
                    data.iloc[i, j] = 3
    data.to_csv(projectPath + '/res/data.txt', index=None, header=None)
    