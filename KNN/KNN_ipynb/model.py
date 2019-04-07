# coding:utf8
from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


# %matplotlib inline
"""
plt.rcParams['figure.figsize'] = (32.0, 32.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

"""

def test():
    print("fuck")

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
    return X, Y
def load_CIFAR10(dir):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join('datasets', dir, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join('datasets', dir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def distance(X_test, X_train):
    """
    输入:
    X_test -- 由numpy数组表示的测试集，大小为(d , num_test)
    X_train -- 由numpy数组表示的训练集，大小为(d, num_train)
    输出:
    distances -- 测试数据与各个训练数据之间的距离,大小为(num_test, num_train)的numpy数组
    """
    num_test = X_test.shape[1]
    num_train = X_train.shape[1]
    distances = np.zeros((num_test, num_train)) # test和train对应的数组
    # (X_test - X_train)*(X_test - X_train) = -2X_test*X_train + X_test*X_test + X_train*X_train
    #展开平方差公式，是不是这样就可以使用numpy的并行计算？
    #print(X_test.shape,X_train.shape)
    
    dist1 = np.multiply(np.dot(X_test.T,X_train), -2)    # -2X_test*X_train, shape (num_test, num_train)
    dist2 = np.sum(np.square(X_test.T), axis=1, keepdims=True)    # X_test*X_test, shape (num_test, 1)
    dist3 = np.sum(np.square(X_train), axis=0,keepdims=True)    # X_train*X_train, shape(1, num_train)
    distances = np.sqrt(dist1 + dist2 + dist3)

    return distances

def predict(X_test, X_train, Y_train, k = [1]):
    """ 
    输入:
    X_test -- 由numpy数组表示的测试集，大小为(图片长度 * 图片高度 * 3 , 测试样本数)
    X_train -- 由numpy数组表示的训练集，大小为(图片长度 * 图片高度 * 3 , 训练样本数)
    Y_train -- 由numpy数组（向量）表示的训练标签，大小为 (1, 训练样本数)
    k -- 选取与训练集最近邻的数量的list
    输出:
    Y_prediction -- 包含X_test中所有预测值的numpy数组（向量）
    distances -- 由numpy数组表示的测试数据与各个训练数据之间的距离,大小为(测试样本数, 训练样本数)
    """
    
    distances = distance(X_test, X_train)
    
#     print(k)
    num_test = X_test.shape[1]
    Y_prediction = np.zeros((num_test,len(k)))
    for i in range(num_test):
        for j,item_k in enumerate(k):
            dists_min_k = np.argsort(distances[i])[:item_k]     # 按照距离递增次序进行排序,选取距离最小的k个点 
            y_labels_k = Y_train[0,dists_min_k]     # 确定前k个点的所在类别
            Y_prediction[i][j] = np.argmax(np.bincount(y_labels_k)) # 返回前k个点中出现频率最高的类别作为测试数据的预测分类

    
    
#     print(Y_prediction)
    return Y_prediction


def model(X_test, Y_test, X_train, Y_train, k = [1], print_correct = False):
    """
    输入：
    X_test -- 由numpy数组表示的测试集，大小为(图片长度 * 图片高度 * 3 , 测试样本数)
    X_train -- 由numpy数组表示的训练集，大小为(图片长度 * 图片高度 * 3 , 训练样本数)
    Y_train -- 由numpy数组（向量）表示的训练标签，大小为 (1, 训练样本数)
    Y_test -- 由numpy数组（向量）表示的测试标签，大小为 (1, 测试样本数)
    k -- 选取与训练集最近邻的数量的系列数组
    print_correct -- 设置为true时，打印正确率
    输出：
    d -- 包含模型信息的字典的数组
    """
    Y_prediction= predict(X_test, X_train, Y_train, k)
    num_correct = np.sum(Y_prediction == Y_test)
    d_array=[]
    for i,k_item in enumerate(k):
        accuracy = np.mean(Y_prediction[:,i] == Y_test)
        if print_correct:
            print('Correct %d/%d: The test accuracy: %f' % (num_correct, X_test.shape[1], accuracy))

        d_array.append({"k": k,
             "Y_prediction": Y_prediction, 
    #          "distances" : distances,
             "accuracy": accuracy})
    #print(d_array)
    #安装k数组里面的元素的顺序，排列成d
    return d_array
