#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:11:24 2019

@author: 贾恒
"""
import os
import time

import nltk
import pandas as pd
import re

projectPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
webshellCount = 0
nonwebshellCount = 0


def webshellWordFreq():
    global webshellCount
    startTime = time.time()
    webshellDict = {}
    webshellPath = projectPath + "/res/samples/" + "webshell/"
    for i in os.walk(webshellPath):
        for j in i[2]:
            if j.endswith(r".php"):
                webshellCount += 1
                with open(i[0] + '/' + j, "r", errors="ignore") as f:
                    source = f.read()
                    # 去/*...*/注释
                    source = re.compile("\/\*[\s\S]*\*\/").sub('', source)
                    # 去//注释
                    source = re.compile("\/\/.*?").sub('', source)
                    words = re.findall("[a-zA-Z]+", source)
                    # print([word.lower() for word in words])
                    freqDist = nltk.FreqDist([word.lower() for word in words])
                    for key in freqDist.keys():
                        if key in webshellDict.keys():
                            webshellDict[key] += freqDist[key]
                        else:
                            webshellDict[key] = freqDist[key]
                if webshellCount % 20 == 0:
                    print("Processed: %d webshells (Elapsed Time: %.3f)" % (webshellCount, time.time() - startTime))
    return sorted(webshellDict.items(), key=lambda x: x[1], reverse=True)


def nonwebshellWordFreq():
    global nonwebshellCount
    startTime = time.time()
    nonwebshellDict = {}
    nonwebshellPath = projectPath + "/res/samples/" + "nonwebshell/"
    for i in os.walk(nonwebshellPath):
        for j in i[2]:
            if j.endswith(r".php"):
                nonwebshellCount += 1
                with open(i[0] + '/' + j, "r", errors="ignore") as f:
                    source = f.read()
                    # 去/*...*/注释
                    source = re.compile("\/\*[\s\S]*\*\/").sub('', source)
                    # 去//注释
                    source = re.compile("\/\/.*?").sub('', source)
                    words = re.findall("[a-zA-Z]+", source)
                    # print([word.lower() for word in words])
                    freqDist = nltk.FreqDist([word.lower() for word in words])
                    for key in freqDist.keys():
                        if key in nonwebshellDict.keys():
                            nonwebshellDict[key] += freqDist[key]
                        else:
                            nonwebshellDict[key] = freqDist[key]
                if nonwebshellCount % 20 == 0:
                    print("Processed: %d nonwebshells (Elapsed Time: %.3f)" % (
                    nonwebshellCount, time.time() - startTime))
    return sorted(nonwebshellDict.items(), key=lambda x: x[1], reverse=True)


def main():
    startTime = time.time()
    pd.DataFrame(webshellWordFreq()).to_csv(projectPath + "/res/keyWordsWebshell.csv", header=False, 
                                            index=False, mode="w", encoding="utf-8")
    pd.DataFrame(nonwebshellWordFreq()).to_csv(projectPath + "/res/keyWordsNonwebshell.csv", header=False, 
                                               index=False, mode="w", encoding="utf-8")
    endTime = time.time()
    print("Total:")
    print("Analyze WORD FREQUENCY successfully in  %.3f  s" % (endTime - startTime))
    print("  " + str(webshellCount) + "  webshell examples")
    print("  " + str(nonwebshellCount) + "  nonwebshell examples")


if __name__ == "__main__":
    main()
