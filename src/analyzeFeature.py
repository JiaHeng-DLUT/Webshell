# -*- coding: utf-8 -*-
"""
@author: 贾恒
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

# 项目所在路径
projectPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def main():
    feature = pd.read_csv(projectPath+'/res/feature.csv')
    x = feature.iloc[:, 6]
    plt.plot(x)
    plt.show()
    print(x.count())
    print(x.max())
    print(x.min())
    print(x.mean())
    print(x.median())
    print(x.std())
        
    
if __name__ == "__main__":
    main()
    