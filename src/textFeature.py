# -*- coding: utf-8 -*-
"""
@author: 贾恒
"""
import os
import re
import time

import numpy as np
import pandas as pd


projectPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def extractTextFeature(sample, webshell):
    # Extract TEXT FEATURE from sample
    # TEXT FEATURE
    f = {
        "fileName": sample,
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
        # 不同单词数量
        f["diffWordsNum"] = len(set(words))
        # 最大单词长度
        if words:
            f["longestWordLen"] = max([len(word) for word in words])
        # 字符数量
        f["totalCharNum"] = len(re.findall("\S", source))
        # 特殊字符数量
        f["totalSpecialCharNum"] = f["totalCharNum"] - \
            len(re.findall("[a-zA-Z0-9]", source))
        f["webshell"] = webshell
    return f


def extractFeature():
    # init
    startTime = time.time()
    textFeature = pd.DataFrame(columns=['fileName', 'commentCharNum', 
                                        'wordsNum', 'diffWordsNum', 
                                        'longestWordLen', 'totalCharNum', 
                                        'totalSpecialCharNum', 'webshell'])
    webshellPath = projectPath + "/res/samples/" + "webshell/"
    nonwebshellPath = projectPath + "/res/samples/" + "nonwebshell/"
    webshell = os.listdir(webshellPath)
    # webshelll
    webshellCount = 0
    for _ in webshell:
        webshellCount += 1
        textFeature = textFeature.append(extractTextFeature(webshellPath+_, 1), 
                                         ignore_index=True)
    # nonwebshell    
    nonwebshellCount = 0
    for i in os.walk(nonwebshellPath):
        for j in i[2]:
            if j.endswith(r".php"):
                nonwebshellCount += 1
                textFeature = textFeature.append(extractTextFeature(i[0]+'/'+j, 0), 
                                         ignore_index=True)
    # write into file
    if not os.path.exists(projectPath + "/res/"):
        os.makedirs(projectPath + "/res/")
    textFeature.to_csv(projectPath + "/res/textFeature.csv", header=True, 
                       index=False, encoding="utf-8")
    # output
    endTime = time.time()
    print("提取样本特征成功！耗时：%.3f s" % (endTime - startTime))
    print("   " + str(webshellCount) + " 个webshell样本")
    print("   " + str(nonwebshellCount) + " 个非webshell样本")


def clean_data():
    with open(projectPath + "/res/feature.csv") as file:
        f = pd.read_csv(file, header=None, encoding="utf-8")

        # 端点1, 端点2
        (p1, p2) = ([], [])
        for _ in f.columns:
            # 众数
            mean = f[_].mean()
            # 最大值
            maximum = f[_].max()
            p1.append(mean / 2)
            p2.append((maximum - mean) / 2)

        data = np.zeros((f.shape[0], 3), dtype=int)
        for i in range(0, f.shape[0]):
            # text feature
            s = ""
            for j in range(0, 6):
                if f.at[i, j] < p1[j]:
                    s += "0"
                elif p1[j] <= f.at[i, j] <= p2[j]:
                    s += "1"
                elif f.at[i, j] > p2[j]:
                    s += "2"
                data[i][0] = int(s, 3)
            # function feature
            s = ""
            for j in range(6, 12):
                if f.at[i, j] < p1[j]:
                    s += "0"
                elif p1[j] <= f.at[i, j] <= p2[j]:
                    s += "1"
                elif f.at[i, j] > p2[j]:
                    s += "2"
                data[i][1] = int(s, 3)
            # webshell
            data[i][2] = 2 * f.at[i, 12] - 1

        pd.DataFrame(data).to_csv(
            projectPath + '/res/data.csv',
            header=False,
            index=False,
            mode="w",
            encoding="utf-8")


def main():
    extractFeature()
    # clean_data()
    
    
if __name__ == "__main__":
    main()
    