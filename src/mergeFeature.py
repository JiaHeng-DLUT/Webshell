# -*- coding: utf-8 -*-
"""
@author: 贾恒
"""
import os

import pandas as pd

# 项目所在路径
projectPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def main():
    neopiWebshell = pd.read_csv(projectPath+'/res/neopiWebshell.csv')
    neopiNonwebshell = pd.read_csv(projectPath+'/res/neopiNonwebshell.csv')
    neopiFeature = pd.concat([neopiWebshell, neopiNonwebshell], axis=0)
    neopiFeature.to_csv(projectPath+'/res/neopiFeature.csv',index=None)
    textFeature = pd.read_csv(projectPath+'/res/textFeature.csv')
    keyWordsFeature = pd.read_csv(projectPath+'/res/keyWordsFeature.csv')
    
    feature = neopiFeature.iloc[:,:-2].merge(keyWordsFeature, on='filename')
    feature = feature.merge(textFeature, on='filename')
    feature.iloc[:, 1:].to_csv(projectPath+'/res/feature.csv',index=None)
    
    
if __name__ == "__main__":
    main()
    