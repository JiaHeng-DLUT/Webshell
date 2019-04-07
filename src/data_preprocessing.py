# -*- coding: utf-8 -*-
"""
@author: 贾恒
"""
import os
import re
import time

import numpy as np
import pandas as pd
import requests

# 项目所在路径
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_php_fn():
    # 从w3school教程网站爬取php函数并写入csv文件
    start_time = time.time()
    temp = ["array", "filesystem", "math", "xml"]
    for _ in temp:
        res = requests.get("http://www.w3school.com.cn/php/php_ref_" + _ +
                           ".asp")
        res.encoding = "gbk"
        fns = re.findall(".asp\">[a-zA-Z0-9_]*", res.text)
        for i in range(0, len(fns)):
            fns[i] = fns[i].split(".asp\">")[1]
        if not os.path.exists(project_path + "/res/php/"):
            os.makedirs(project_path + "/res/php/")
        pd.DataFrame(fns[:-2]).to_csv(
            project_path + "/res/php/" + _ + ".csv",
            header=False,
            index=False,
            encoding="utf-8")
    end_time = time.time()
    print("爬取php函数成功！耗时：%.3f s" % (end_time - start_time))


def extract_feature_single(sample, webshell, fn_files, fn):
    # 从sample里面提取文本特征和函数特征
    # 文本特征
    f = {
        "cmt_chars_num": 0,
        "words_num": 0,
        "diff_words_num": 0,
        "longest_word_len": 0,
        "chars_num": 0,
        "special_chars_num": 0
    }
    # 函数特征
    for fn_file in fn_files:
        f[str(fn_file[:-4])] = 0
    # 从样本文件中提取特征
    with open(sample, "r", errors="ignore") as file:
        source = file.read()
        # 注释字符数
        f["cmt_chars_num"] = len(source)
        # 去/*...*/注释
        source = re.compile("\/\*[\s\S]*\*\/").sub('', source)
        # 去//注释
        source = re.compile("\/\/.*?").sub('', source)
        # 字符串中的注释暂时当作注释，因为正常的代码中字符串极少包含注释
        f["cmt_chars_num"] -= len(source)
        words = re.findall("[a-zA-Z]+", source)
        # 单词数量
        f["words_num"] = len(words)
        # 不同单词数量
        f["diff_words_num"] = len(set(words))
        # 最大单词长度
        if words:
            f["longest_word_len"] = max([len(word) for word in words])
        # 字符数量
        f["chars_num"] = len(re.findall("\S", source))
        # 特殊字符数量
        f["special_chars_num"] = f["chars_num"] - \
            len(re.findall("[a-zA-Z0-9]", source))
        for i in range(0, len(fn)):
            for item in fn[i]:
                f[str(fn.index[i])] += len(re.findall(item, source))
        f["webshell"] = webshell
        if not os.path.exists(project_path + "/res/"):
            os.makedirs(project_path + "/res/")
        pd.DataFrame(
            f, index=[0]).to_csv(
                project_path + "/res/feature.csv",
                header=False,
                index=False,
                mode="a",
                encoding="utf-8")


def extract_feature():
    start_time = time.time()
    fn_files = os.listdir(project_path + "/res/php/")
    if('.DS_Store' in fn_files):
        fn_files.remove('.DS_Store')
    fn = pd.Series([[], [], [], []],
                   index=[fn_file[:-4] for fn_file in fn_files])
    for fn_file in fn_files:
        # 使用pandas直接读取文件报错
        # OSError: Initializing from file failed
        # 一般是因为文件名中带有中文
        # 可做如下处理
        with open(project_path + "/res/php/" + fn_file) as f:
            csv_data = pd.read_csv(f)
            fn[str(fn_file[:-4])] = csv_data
    samples_path = project_path + "/res/samples/"
    webshell = os.listdir(samples_path + "webshell/")
    nonwebshell_path = samples_path + "nonwebshell/"
    if os.path.exists(project_path + "/res/feature.csv"):
        with open(project_path + "/res/feature.csv", "w") as f:
            f.write("")
    webshell_count = 0
    for _ in webshell:
        webshell_count += 1
        extract_feature_single(samples_path + "webshell/" + _, 1, fn_files, fn)
    nonwebshell_count = 0
    for i in os.walk(nonwebshell_path):
        for j in i[2]:
            if j.endswith(r".php"):
                nonwebshell_count += 1
                extract_feature_single(os.path.join(i[0], j), 0, fn_files, fn)
    end_time = time.time()
    print("提取样本特征成功！耗时：%.3f s" % (end_time - start_time))
    print("   " + str(webshell_count) + " 个webshell样本")
    print("   " + str(nonwebshell_count) + " 个非webshell样本")


def clean_data():
    with open(project_path + "/res/feature.csv") as file:
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
            project_path + '/res/data.csv',
            header=False,
            index=False,
            mode="w",
            encoding="utf-8")


def main():
    # 第零步：------------------------准备工作
    get_php_fn()
    # 第一步：------------------------数据预处理
    extract_feature()
    # 第二步：------------------------数据清洗
    # clean_data()
    
    
if __name__ == "__main__":
    main()
    