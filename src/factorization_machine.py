import os
from random import normalvariate  # 正态分布

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# the number of features
features_num = 2


def load_data(csv_file):
    """
    @function: 从csv文件中导入数据
    @args: 
        csv_file(string)  csv file name
    @returns: 
        data(np.ndarray)  feature vector
        label(np.ndarray)  indacator of webshell, 1: webshell, -1: nonwebshell
    """
    with open(csv_file) as f:
        csv_data = pd.read_csv(f, header=None, encoding="utf-8")
        data = np.zeros((csv_data.shape[0], features_num))
        label = np.zeros(csv_data.shape[0])
        for row, index in csv_data.iterrows():
            data[row] = csv_data.values[row, :-1]
            label[row] = csv_data.values[row, -1]
        return data, label


def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


def initialize_v(n, k):
    """
    @function: 初始化交叉项
    @args:
        n(int)  特征的个数
        k(int)  FM模型的超参数
    @returns: 
        v(mat)  交叉项的系数权重
    """
    v = np.mat(np.zeros((n, k)))
    for i in range(n):
        for j in range(k):
            # 利用正态分布生成每一个权重
            v[i, j] = normalvariate(0, 0.2)
    return v


def stochastic_gradient_descent(data_train, class_labels_train, k, max_iter, alpha):
    """
    @function: 利用随机梯度下降法训练FM模型
    @args:  
        data_train(np.mat)  特征
        class_labels_train(np.mat)  类别
        k(int)  v的维数
        max_iter(int)  最大迭代次数
        alpha(float)  学习率
    @returns: 
        w0(float)
        w(mat)  
        v(mat)  权重
    """
    m, n = np.shape(data_train)

    # 1、初始化参数
    w = np.zeros((n, 1))  # 其中n是特征的个数
    w0 = 0  # 偏置项
    v = initialize_v(n, k)  # 初始化V

    # 2、训练
    for it in range(0, max_iter):
        for x in range(0, m):  # 随机优化，对每一个样本而言的
            inter_1 = data_train[x] * v
            inter_2 = np.multiply(data_train[x], data_train[x]) * \
                np.multiply(v, v)  # multiply对应元素相乘
            # 完成交叉项
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            p = w0 + data_train[x] * w + interaction  # 计算预测的输出
            loss = sigmoid(class_labels_train[x] * p[0, 0]) - 1

            w0 = w0 - alpha * loss * class_labels_train[x]
            for i in range(n):
                if data_train[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * \
                        class_labels_train[x] * data_train[x, i]

                    for j in range(k):
                        v[i, j] = v[i, j] - alpha * loss * class_labels_train[x] * \
                            (data_train[x, i] * inter_1[0, j] -
                             v[i, j] * data_train[x, i] * data_train[x, i])

        # 计算损失函数的值
        if it % 1000 == 0:
            print("\t------- iter: ", it, " , cost: ",
                  get_cost(get_prediction(np.mat(data_train), w0, w, v), class_labels_train))

    # 3、返回最终的FM模型的参数
    return w0, w, v


def get_cost(predict, class_labels):
    """
    @function: 计算预测准确性
    @args:  
        predict(list)  预测值
        class_labels(list)  标签
    @returns:
        error(float)  计算损失函数的值
    """
    m = len(predict)
    error = 0.0
    for i in range(m):
        error -= np.log(sigmoid(predict[i] * class_labels[i]))
    return error


def test_get_cost():
    assert get_cost([1],[1])==0

def get_prediction(data_matrix, w0, w, v):
    """
    @function: 得到预测值
    @args:  
        data_matrix(mat)  特征
        w(int)  常数项权重
        w0(int)  一次项权重
        v(float)  交叉项权重
    @returns: 
        result(list)  预测的结果
    """
    m = np.shape(data_matrix)[0]
    result = []
    for x in range(m):
        inter_1 = data_matrix[x] * v
        inter_2 = np.multiply(data_matrix[x], data_matrix[x]) * \
            np.multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
        p = w0 + data_matrix[x] * w + interaction  # 计算预测的输出
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result


def get_accuracy(predict, class_labels):
    """
    @function: 计算预测准确性
    @args:  
        predict(list)  预测值
        class_labels(list)  标签
    @returns:
        float(error)/all_item(float)  错误率
    """
    m = len(predict)
    all_item = 0
    error = 0
    for i in range(m):
        all_item += 1
        if float(predict[i]) < 0.5 and class_labels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and class_labels[i] == -1.0:
            error += 1
        else:
            continue
    return float(error) / all_item


def save_model(file_name, w0, w, v):
    """
    @function: 保存训练好的FM模型
    @args:  
        file_name(string):保存的文件名
        w0(float):偏置项
        w(mat):一次项的权重
        v(mat):交叉项的权重
    """
    with open(file_name, "w") as f:
        # 1、保存w0
        f.write(str(w0) + "\n")
        # 2、保存一次项的权重
        w_array = []
        m = np.shape(w)[0]
        for i in range(m):
            w_array.append(str(w[i, 0]))
        f.write("\t".join(w_array) + "\n")
        # 3、保存交叉项的权重
        m1, n1 = np.shape(v)
        for i in range(m1):
            v_tmp = []
            for j in range(n1):
                v_tmp.append(str(v[i, j]))
            f.write("\t".join(v_tmp) + "\n")


def main():
    project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print("---------- 1.load data ----------")
    # 1、导入数据
    data, label = load_data(project_path+"/res/data.csv")
    print("---------- 2.divide data ----------")
    # 2、划分训练集和预测集
    data_train, data_test, class_labels_train, class_labels_test = train_test_split(data, label, test_size=0.2, random_state=42)
    print("---------- 3.learning ----------")
    # 3、利用随机梯度训练FM模型
    w0, w, v = stochastic_gradient_descent(np.mat(data_train), class_labels_train, 2, 10000, 0.0000005)
    predict_result = get_prediction(np.mat(data_train), w0, w, v)  # 得到训练的准确性
    print("----------training accuracy: %f" % (1 - get_accuracy(predict_result, class_labels_train)))
    print("---------- 4.save result ---------")
    # 4、保存训练好的FM模型
    save_model(project_path+r"/res/weights", w0, w, v)


if __name__ == "__main__":
    main()
