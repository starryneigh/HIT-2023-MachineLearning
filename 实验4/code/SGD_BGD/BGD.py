# --*-- coding:utf-8 --*--
"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-19
"""
import numpy as np
from matplotlib import pyplot as plt
from functions import *


class NN:
    def __init__(self, ni, nh, no, batch=16):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni + 1  # 增加一个偏差节点
        self.nh = nh
        self.no = no
        self.batch = batch

        # 激活神经网络的所有节点
        self.x = np.ones((self.batch, self.ni))
        # print(self.x)
        self.h1 = np.ones((self.batch, self.nh))
        self.h2 = np.ones((self.batch, self.no))

        # 建立权重（矩阵）, 设为随机值
        self.w1 = np.random.randn(self.ni, self.nh)
        self.w2 = np.random.randn(self.nh, self.no)

    def forward(self, input_x):   # input_x: shape(batch, ni)
        # print(input_x)
        self.x = np.hstack((input_x, np.ones((self.batch, 1))))
        z1 = np.dot(self.x, self.w1)
        self.h1 = sigmoid(z1)
        z2 = np.dot(self.h1, self.w2)
        self.h2 = softmax(z2)
        return self.h2  # h2: shape(batch, no)

    def backward(self, input_y, alpha):
        z2_grad = self.h2 - input_y
        w2_grad = np.matmul(self.h1.reshape(self.batch, self.nh, 1),
                         z2_grad.reshape(self.batch, 1, self.no)).mean(axis=0)

        h1_grad = np.dot(z2_grad, self.w2.T)
        z1_grad = h1_grad * dsigmoid(self.h1)
        w1_grad = np.matmul(self.x.reshape(self.batch, self.ni, 1),
                         z1_grad.reshape(self.batch, 1, self.nh)).mean(axis=0)
        # print(w1_grad.shape)

        self.w2 -= alpha * w2_grad
        self.w1 -= alpha * w1_grad

        err = loss(self.h2, input_y).mean(axis=0)
        return err

    def train(self, x_train, y_train, epoch=100, alpha=0.03, threshold=1e-3, use_thr=True):
        num = x_train.shape[0]
        batch_num = num // self.batch
        pre_err = 0
        for i in range(epoch):
            err = 0
            for j in range(batch_num):
                self.forward(x_train[j*self.batch:(j+1)*self.batch])
                err += self.backward(y_train[j*self.batch:(j+1)*self.batch], alpha)
            err = err / batch_num
            if use_thr:
                if abs(pre_err - err) <= threshold:
                    print(f'iter = {i+1},\terr = {err}, diff = {err - pre_err}')
                    return i
            if i % 100 == 0 or i==epoch-1:
                print(f'iter = {i+1},\terr = {err}, diff = {err - pre_err}')
            pre_err = err
        return epoch

    def test(self, x_test, y_test):
        cnt = 0
        num = x_test.shape[0]
        batch_num = num // self.batch
        predicts = []
        for i in range(batch_num):
            predict = self.forward(x_test[i*self.batch:(i+1)*self.batch])
            predict = np.argmax(predict, axis=1)
            for j in range(self.batch):
                predicts.append(predict[j])
                if y_test[i*self.batch + j][predict[j]] == 1:
                    cnt += 1
        predicts = np.array(predicts)
        return round(cnt/num, 3), predicts


def predict_show(predicts, x_test, y_test, dic):
    num = predicts.shape[0]
    k = y_test.shape[1]
    y = np.argmax(y_test, axis=1)
    # print(y)
    predict_label = []
    true_label = []
    clusters = []
    for i in range(k):
        clusters.append([])
    for i in range(num):
        for key, value in dic.items():
            if value == y[i]:
                true_label.append(key)
            if value == predicts[i]:
                predict_label.append(key)
                clusters[predicts[i]].append(x_test[i])
    for i in range(num):
        print(f'第{i}项的真实标签为：{true_label[i]}，预测标签为：{predict_label[i]}', end='\t')
    print()
    for i in range(k):
        cluster = np.array(clusters[i])
        plt.scatter(cluster[:, 0], cluster[:, 1], label=i, marker='.')
    plt.legend()
    plt.show()
    return predict_label


