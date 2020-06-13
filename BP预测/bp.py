#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:54:44 2017

@author: zhangshichaochina@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

def sigmoid(z):
    """
    使用numpy实现sigmoid函数
    
    参数：
    Z numpy array
    输出：
    A 激活值（维数和Z完全相同）
    """
    return 1/(1 + np.exp(-z))

def relu(z):
    """
    线性修正函数relu
    
    参数：
    z numpy array
    输出：
    A 激活值（维数和Z完全相同）
    
    """
    return np.array(z>0)*z

def sigmoidBackward(dA, cacheA):
    """
    sigmoid的反向传播
    
    参数：
    dA 同层激活值
    cacheA 同层线性输出
    输出：
    dZ 梯度
    
    """
    s = sigmoid(cacheA)
    diff = s*(1 - s)
    dZ = dA * diff
    return dZ

def reluBackward(dA, cacheA):
    """
    relu的反向传播
    
    参数：
    dA 同层激活值
    cacheA 同层线性输出
    输出：
    dZ 梯度
    
    """
    Z = cacheA
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    return dZ

def iniPara(laydims):
    """
    随机初始化网络参数
    
    参数：
    laydims 一个python list
    输出：
    parameters 随机初始化的参数字典（”W1“，”b1“，”W2“，”b2“, ...）
    """
    np.random.seed(1)
    parameters = {}
    for i in range(1, len(laydims)):
        parameters['W'+str(i)] = np.random.randn(laydims[i], laydims[i-1])/ np.sqrt(laydims[i-1])
        parameters['b'+str(i)] = np.zeros((laydims[i], 1))
    return parameters


def loadData(dataDir):
    """
    导入数据
    
    参数：
    dataDir 数据集路径
    输出：
    训练集，测试集以及标签
    """
    train_dataset = h5py.File(dataDir+'/train.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(dataDir+'/test.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def forwardLinear(W, b, A_prev):
    """
    前向传播
    """
    Z = np.dot(W, A_prev) + b
    cache = (W, A_prev, b)
    return Z, cache

def forwardLinearActivation(W, b, A_prev, activation):
    """
    带激活函数的前向传播
    """
    Z, cacheL = forwardLinear(W, b, A_prev)
    cacheA = Z
    if activation == 'sigmoid':
        A = sigmoid(Z)
    if activation == 'relu':
        A = relu(Z)
    cache = (cacheL, cacheA)
    return A, cache

def forwardModel(X, parameters):
    """
    完整的前向传播过程
    """
    layerdim = len(parameters)//2
    caches = []
    A_prev = X
    for i in range(1, layerdim):
        A_prev, cache = forwardLinearActivation(parameters['W'+str(i)], parameters['b'+str(i)], A_prev, 'relu')
        caches.append(cache)
        
    AL, cache = forwardLinearActivation(parameters['W'+str(layerdim)], parameters['b'+str(layerdim)], A_prev, 'sigmoid')
    caches.append(cache)
    
    return AL, caches

def computeCost(AL, Y):
    """
    代价函数的计算
    
    参数：
    AL 输出层的激活输出
    Y 标签值
    输出：
    cost 代价函数值
    """
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    return cost

def linearBackward(dZ, cache):
    """
    线性部分的反向传播
    
    参数：
    dZ 当前层误差
    cache （W, A_prev, b）元组
    输出：
    dA_prev 上一层激活的梯度
    dW 当前层W的梯度
    db 当前层b的梯度
    """
    W, A_prev, b = cache
    m = A_prev.shape[1]
    
    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linearActivationBackward(dA, cache, activation):
    """
    非线性部分的反向传播
    
    参数：
    dA 当前层激活输出的梯度
    cache （W, A_prev, b）元组
    activation 激活函数类型
    输出：
    dA_prev 上一层激活的梯度
    dW 当前层W的梯度
    db 当前层b的梯度
    """
    cacheL, cacheA = cache
    
    if activation == 'relu':
        dZ = reluBackward(dA, cacheA)
        dA_prev, dW, db = linearBackward(dZ, cacheL)
    elif activation == 'sigmoid':
        dZ = sigmoidBackward(dA, cacheA)
        dA_prev, dW, db = linearBackward(dZ, cacheL)
    
    return dA_prev, dW, db

def backwardModel(AL, Y, caches):
    """
    完整的反向传播过程
    
    参数：
    AL 输出层结果
    Y 标签值
    caches 【cacheL, cacheA】
    输出：
    diffs 梯度字典
    """
    layerdim = len(caches)
    Y = Y.reshape(AL.shape)
    L = layerdim
    
    diffs = {}
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    currentCache = caches[L-1]
    dA_prev, dW, db =  linearActivationBackward(dAL, currentCache, 'sigmoid')
    diffs['dA' + str(L)], diffs['dW'+str(L)], diffs['db'+str(L)] = dA_prev, dW, db
    
    for l in reversed(range(L-1)):
        currentCache = caches[l]
        dA_prev, dW, db =  linearActivationBackward(dA_prev, currentCache, 'relu')
        diffs['dA' + str(l+1)], diffs['dW'+str(l+1)], diffs['db'+str(l+1)] = dA_prev, dW, db
        
    return diffs


def updateParameters(parameters, diffs, learningRate):
    """
    更新参数
    
    参数：
    parameters 待更新网络参数
    diffs 梯度字典
    learningRate 学习率
    输出：
    parameters 网络参数字典
    """
    layerdim = len(parameters)//2
    for i in range(1, layerdim+1):
        parameters['W'+str(i)]-=learningRate*diffs['dW'+str(i)]
        parameters['b'+str(i)]-=learningRate*diffs['db'+str(i)]
    return parameters

def finalModel(X, Y, layerdims, learningRate=0.01, numIters=5000,pringCost=False):
    """
    最终的BP神经网络模型
    
    参数：
    X 训练集特征
    Y 训练集标签
    layerdims 一个明确网络结构的python list
    learningRate 学习率
    numIters 迭代次数
    pringCost 打印标志
    输出：
    parameters 参数字典
    """
    np.random.seed(1)
    costs = []
    
    parameters = iniPara(layerdims)
    
    for i in range(0, numIters):
        AL, caches = forwardModel(X, parameters)
        cost = computeCost(AL, Y)
        
        diffs = backwardModel(AL,Y, caches)
        parameters = updateParameters(parameters,diffs, learningRate)
    
        if pringCost and i%100 == 0:
            print(cost)
            costs.append(np.sum(cost))
    return parameters

def predict(X, y, parameters):
    """
    在测试集上预测
    
    参数：
    X 输入特征值
    y 测试集标签
    parameters 参数字典
    输出：
    p 预测得到的标签
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 
    p = npc0 .zeros((1,m))
    probas, caches = forwardModel(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p, np.sum((p == y)/m)
