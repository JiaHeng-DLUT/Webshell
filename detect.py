#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:50:23 2019

@author: jiaheng
"""

import os

import numpy as np
import pandas as pd
import re
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.svm import SVC

projectPath = os.path.dirname(os.path.realpath(__file__))
keyWords = 0
with open(projectPath + "/res/keyWordsUsed.csv", "r") as f:
    keyWords = pd.read_csv(f, header=None)
keyWords = keyWords[0].tolist()


def extractTextFeature(sample, webshell):
    # Extract TEXT FEATURE from sample
    # TEXT FEATURE
    f = {
        "filename": sample,
        "commentCharNum": 0,
        "wordsNum": 0,
        "diffWordsNum": 0,
        "longestWordLen": 0,
        "totalCharNum": 0,
        "totalSpecialCharNum": 0
    }
    with open(sample, "r", errors="ignore") as file:
        source = file.read()
        # 注释字符数
        f["commentCharNum"] = len(source)
        # 去/*...*/注释
        source = re.compile("\/\*[\s\S]*\*\/").sub('', source)
        # 去//注释
        source = re.compile("\/\/.*?").sub('', source)
        # 字符串中的注释暂时当作注释，因为正常的代码中字符串极少包含注释
        f["commentCharNum"] -= len(source)
        words = re.findall("[a-zA-Z]+", source)
        # 单词数量
        f["wordsNum"] = len(words)
        diffWords = set(word.lower() for word in words)
        # 不同单词数量
        f["diffWordsNum"] = len(diffWords)
        # 最大单词长度
        if words:
            f["longestWordLen"] = max([len(word) for word in diffWords])
        # 字符数量
        f["totalCharNum"] = len(re.findall("\S", source))
        # 特殊字符数量
        f["totalSpecialCharNum"] = f["totalCharNum"] - len(re.findall("[a-zA-Z0-9]", source))
        f["webshell"] = webshell
    # print(f)
    return f


def extractKeyWordsFeature(sample):
    # Extract KEY WORDS FEATURE from sample
    # KEY WORDS FEATURE
    f = dict.fromkeys(keyWords, 0)
    with open(sample, "r", errors="ignore") as file:
        source = file.read()
        # 去/*...*/注释
        source = re.compile("\/\*[\s\S]*\*\/").sub('', source)
        # 去//注释
        source = re.compile("\/\/.*?").sub('', source)
        # 字符串中的注释暂时当作注释，因为正常的代码中字符串极少包含注释
        words = re.findall("[a-zA-Z]+", source)
        words = [word.lower() for word in words]
        for _ in keyWords:
            f[_] = words.count(_)
    f["filename"] = sample
    # print(f)
    return f


# extract
textFeature = pd.DataFrame(
    columns=["filename", "commentCharNum", "wordsNum", "diffWordsNum", "longestWordLen", "totalCharNum",
             "totalSpecialCharNum", "webshell"])
keyWordsFeature = pd.DataFrame(columns=keyWords)
for i in os.walk(projectPath+"/samples2detect"):
    for j in i[2]:
        if j.endswith(r".php"):
            textFeature = textFeature.append(extractTextFeature(i[0] + '/' + j, 1), ignore_index=True)
            keyWordsFeature = keyWordsFeature.append(extractKeyWordsFeature(i[0] + '/' + j), ignore_index=True)
# merge
neopiFeature = pd.read_csv(projectPath+'/neopiFeature.csv')
feature = neopiFeature.iloc[:,:-2].merge(keyWordsFeature, on='filename')
feature = feature.merge(textFeature, on='filename')
# data preprocess
p1 = pd.read_csv(projectPath + "/res/median.csv", header=None)
p2 = pd.read_csv(projectPath + "/res/mean.csv", header=None)             
[m, n] = feature.shape
data = feature.copy()
for i in range(0, m):
    for j in range(0, n-2):
        if feature.iloc[i, j+1] <= p1.iloc[j,0]:
            data.iloc[i, j+1] = 1
        else:
            if feature.iloc[i, j+1] <= p2.iloc[j,0]:
                data.iloc[i, j+1] = 2
            else:
                data.iloc[i, j+1] = 3
data.to_csv(projectPath+"/res/data2predict.csv", header=None)
model = joblib.load(projectPath+"/res/model.joblib")
predictedLabel = model.predict(data.iloc[:, 1:-1])
for i in range(0, m):
    if predictedLabel[i]:
        print("  %s  is a webshell\n" % data.iloc[i, 0])
    else:
        print("  %s  is NOT a webshell\n" % data.iloc[i, 0])
