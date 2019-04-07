#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:27:42 2019

@author: jiaheng
"""
# ./libFM -task c -train libfm_train -test libfm_test -iter 2000 -learn_rate 0.05 -method SGD

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


features_num = 10
    

def load_data(csv_file):
    with open(csv_file) as f:
        csv_data = pd.read_csv(f, header=None, encoding="utf-8")
        data = np.zeros((csv_data.shape[0], features_num))
        label = np.zeros(csv_data.shape[0])
        for row, index in csv_data.iterrows():
            data[row] = csv_data.values[row, :-1]
            label[row] = csv_data.values[row, -1]
        return data, label    
    

def convert_data(data, label, file):
    s = ""
    (m, n) = data.shape
    p = [max(data[:,i]) for i in range(n)]
    # print(p)
    p2 = [sum(data[:,i])/m for i in range(n)]
    for i in range(m):
        s += str(int(label[i]))+" "
        for j in range(n):
            if data[i][j]:
                s += str(int(sum(p[0:j-1])+data[i][j]))+":"+str(1)+" "
                # s += str(int((j+data[i][j]/p[j])*p2[j]/features_num))+":"+str(1)+" "
        s += "\n"
    q = [data[i][0] for i in range(m)]
    plt.plot([i for i in range(m)], q)
    with open(file, "w") as f:
        f.write(s)


def main():
    project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print("---------- 1.load data ----------")
    # 1、导入数据
    data, label = load_data(project_path+"/res/feature.csv")
    print("---------- 2.divide data ----------")
    # 2、划分训练集和预测集
    data_train, data_test, labels_train, labels_test = train_test_split(data, label, test_size=0.25, random_state=42)
    print("---------- 3.convert data ----------")
    # 3、转换数据
    convert_data(data_train, labels_train, project_path+"/res/libfm_train")
    convert_data(data_test, labels_test, project_path+"/res/libfm_test")


if __name__ == "__main__":
    main()
