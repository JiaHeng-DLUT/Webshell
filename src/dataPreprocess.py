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


def cleanData():
    with open(projectPath + "/res/feature.csv") as file:
        feature = pd.read_csv(file)
        [m, n] = feature.shape
#        print(n)
        p1 = feature.iloc[:, 1:n-1].median(axis=0)
        p2 = feature.iloc[:, 1:n-1].mean(axis=0)
        print(p1)
        print(p2)
        
        data = feature.copy()       
        for i in range(0, m):
            print("Example: " + str(i))
            for j in range(0, n-1):
                if feature.iloc[i, j] < p1[j-1]:
                    data.iloc[i, j] = 1
                else:
                    if feature.iloc[i, j] < p2[j-1]:
                        data.iloc[i, j] = 2
                    else:
                        data.iloc[i, j] = 3
        data.to_csv(projectPath + '/res/data.txt', index=None, header=None)

#        with open(projectPath + '/res/data.txt', 'w') as f:
#            f.write('')
#        for i in range(0, m):
#            print("Example: " + str(i))
#            data = pd.DataFrame(np.zeros((1,3*(n-2)+1)))
##            print(data.shape)
#            for j in range(1, n-1):
#                t = 3 * (j-1)
#                if feature.iloc[i, j] < p1[j-1]:
#                    data.iloc[0, t+0] = 1
#                else:
#                    if feature.iloc[i, j] < p2[j-1]:
#                        data.iloc[0, t+1] = 2
#                    else:
#                        data.iloc[0, t+2] = 3
#            data.iloc[0, 3*(n-2)] = feature.iloc[i, n-1]
#            data.to_csv(projectPath + '/res/data.txt', mode='a', index=None, header=None)
        

def main():
    cleanData()
    
    
if __name__ == "__main__":
    main()
    