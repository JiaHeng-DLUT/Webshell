#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:50:23 2019

@author: jiaheng
"""

import os
import time

import numpy as np
import pandas as pd
import re

projectPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
keyWordsWebshell = pd.read_csv(projectPath + "/res/keyWordsWebshell.csv")
keyWordsNonwebshell = pd.read_csv(projectPath + "/res/keyWordsNonwebshell.csv")
d = 200
# keyWords = ["filename"]
keyWords1 = set(keyWordsWebshell.iloc[0:d, 0])
keyWords2 = set(keyWordsNonwebshell.iloc[0:d, 0])
keyWords = list(keyWords1.union(keyWords2).difference(keyWords1.intersection(keyWords2)))
with open(projectPath + "/res/keyWordsUsed.csv", "w") as f:
    pd.DataFrame(keyWords).to_csv(f, header=False, index=False)

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


def extractFeature():
    # init
    startTime = time.time()
    textFeature = pd.DataFrame(
        columns=["filename", "commentCharNum", "wordsNum", "diffWordsNum", "longestWordLen", "totalCharNum",
                 "totalSpecialCharNum", "webshell"])
    keyWordsFeature = pd.DataFrame(columns=keyWords)
    webshellPath = projectPath + "/res/samples/" + "webshell/"
    nonwebshellPath = projectPath + "/res/samples/" + "nonwebshell/"
    # webshelll
    webshellCount = 0
    for i in os.walk(webshellPath):
        for j in i[2]:
            if j.endswith(r".php"):
                webshellCount += 1
                textFeature = textFeature.append(extractTextFeature(i[0] + '/' + j, 1), ignore_index=True)
                keyWordsFeature = keyWordsFeature.append(extractKeyWordsFeature(i[0] + '/' + j), ignore_index=True)
                if webshellCount%20 == 0:
                    print("Processed: %d webshells (Elapsed Time: %.3f)" % (webshellCount, time.time()-startTime))
    # nonwebshell
    nonwebshellCount = 0
    for i in os.walk(nonwebshellPath):
        for j in i[2]:
            if j.endswith(r".php"):
                nonwebshellCount += 1
                textFeature = textFeature.append(extractTextFeature(i[0] + '/' + j, 0), ignore_index=True)
                keyWordsFeature = keyWordsFeature.append(extractKeyWordsFeature(i[0] + '/' + j), ignore_index=True)
                if nonwebshellCount%20 == 0:
                    print("Processed: %d nonwebshells (Elapsed Time: %.3f)" % (nonwebshellCount, time.time()-startTime))
    # write into files
    if not os.path.exists(projectPath + "/res/"):
        os.makedirs(projectPath + "/res/")
    textFeature.to_csv(projectPath + "/res/textFeature.csv", header=True, index=False, encoding="utf-8")
    keyWordsFeature.to_csv(projectPath + "/res/keyWordsFeature.csv", header=True, index=False, encoding="utf-8")
    # output
    endTime = time.time()
    print("Total:")
    print("Extract TEXT feature and KEY WORDS feature successfully in  %.3f  s" % (endTime - startTime))
    print("  " + str(webshellCount) + "  webshell examples")
    print("  " + str(nonwebshellCount) + "  nonwebshell examples")


def main():
    extractFeature()


if __name__ == "__main__":
    main()
    