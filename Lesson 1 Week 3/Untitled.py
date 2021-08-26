#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[128]:


# 构建具有单隐层的2类分类神经网络
# 使用具有非线性激活功能激活函数，例如tanh
# 计算交叉熵损失（损失函数）
# 向前和向后传播

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn 
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1) #设置一个固定的随机种子

#将花的图案存储到X,Y
X, Y = load_planar_dataset()

#使用matplotlib可视化数据集，绘制散点图
plt.scatter(X[0, :], X[1, :], c = np.squeeze(Y), s = 40, cmap = plt.cm.Spectral)

#建立模型来适应数据
#X：包含了数据点的数据
#Y：对应X的标签（0红色，1蓝色）
shape_X = X.shape
shape_Y = Y.shape

#训练集里面的数量
m = X.shape[1]

#print("X的维度为: " + str(shape_X))
#print("Y的维度为: " + str(shape_Y))
#print("数据集里面的数据有：" + str(m) + " 个")

#使用sklearn的内置函数来做逻辑回归
#clf = sklearn.linear_model.LogisticRegressionCV()
#clf.fit(X.T, Y.T)

#把逻辑回归分类绘制出来
#plot_decision_boundary(lambda x: clf.predict(x), X, Y)
#plt.title("Logistic Regression") #图标题
#LR_predictions  = clf.predict(X.T) #预测结果
#print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) + 
#    np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
#    "% " + "(正确标记的数据点所占的百分比)")

#1.定义神经网络结构
#2.初始化参数
#3.循环：
#实施前向传播
#计算损失函数
#实现后方传播
#更新参数（梯度下降）

#定义神经网络结构
#n_x:输入层数量
#n_h:隐藏层数量
#n_y:输出层数量

def layer_size(X, Y):
    """
    参数：
    X:输入数据集，维度(输入层数量，样本数量)
    Y:标签，维度(输出层数量，样本数量)
    
    返回值：
    n_x:输入层数量
    n_h:隐藏层数量
    n_y:输出层数量
    """
    n_x = X.shape[0]
    n_h = 4#自己设定
    n_y = Y.shape[0]
    
    return (n_x, n_h, n_y)
    
#测试layer_size
#print("---------测试layer_size----------------")
#X_asses , Y_asses = layer_sizes_test_case()
#(n_x,n_h,n_y) =  layer_size(X_asses,Y_asses)
#print("输入层的节点数量为: n_x = " + str(n_x))
#print("隐藏层的节点数量为: n_h = " + str(n_h))
#print("输出层的节点数量为: n_y = " + str(n_y))

#初始化模型参数
def initialize_parameters(n_x, n_h, n_y):
    """
    参数：
    n_x：输入层节点的数量
    n_h：隐层层节点的数量
    n_y：输出层节点的数量
    返回值：
    parameters:包含参数的字典
        w1：隐藏层权重矩阵，维度(n_h, n_x)
        b1：偏向量，维度(n_h, 1)
        w2：输出层权重，维度(n_y, n_h)
        b2：偏向量，维度(n_y, 1)
    """
    np.random.seed(2)#设定一个随机种子
    W1 = np.random.randn(n_h, n_x) * 0.01 
    b1 = np.zeros(shape = (n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape = (n_y, 1))
    
    #使用断言确保我的数据格式是正确的
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y , 1))
    
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2 }
    
    return parameters

#print("---------测试initialize_parameters----------------")
#n_x , n_h , n_y = initialize_parameters_test_case()
#2,4,1
#parameters = initialize_parameters(n_x, n_h, n_y)
#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))
#print("W2 = " + str(parameters["W2"]))
#print("b2 = " + str(parameters["b2"]))

#循环
#前向传播

#激活函数使用sigmoid()函数或tanh()函数
#实现向前传播，计算Z1,A1,Z2,A2
#反向传播所需的值存储在“cache”中，cache将作为反向传播函数的输入
def forward_propagation(X, parameters):
    """
    参数：
    X：输入数据，维度(n_x,m)
    parameters:初始化的网络参数
    返回值：
    A2：使用sigmoid()函数计算的第二次激活后的数值
    cache：包含A1,Z1，A2，Z2的字典
    """
    W1 = parameters["W1"]
    #(n_h,n_x)
    b1 = parameters["b1"]
    #(n_h,1)
    W2 = parameters["W2"]
    #(n_y,n_h)
    b2 = parameters["b2"]
    #(n_y,1)
    
    #前向传播计算A2
    A0 = X
    Z1 = np.dot(W1 , A0) + b1
    #(n_h,m)
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2 , A1) + b2
    #(n_y,m)
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (b2.shape[0], X.shape[1]))
    
    cache = {"Z1":Z1,
             "A1":A1,
             "Z2":Z2,
             "A2":A2
    }
    
    return (A2, cache)
  
#print("---------测试initialize_parameters----------------")
#X_assess, parameters = forward_propagation_test_case()
#A2, cache = forward_propagation(X_assess, parameters)
#print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))

#计算成本函数
#使用交叉熵损失
#A2等于Y冒
#[]表示层数，()表示样本数
#L(Y冒, Y) = L(A2, Y) = -1/m * sum(y(i) * log(a[2](i)) + (1 - y(i)) * log(1 - a[2](i)))


def compute_cost(A2, Y, parameters):
    """
    参数：
    A2：输出层输出激活值
    Y：训练数据集合对应标签集合
    parameters:神经网络参数的相关字典
    
    返回值：
    cost：交叉熵成本
    """
    m = Y.shape[1]
    #Y(n_y,m)
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    #将每一列加在一起
    cost = np.sum(logprobs, axis = 1, keepdims = True)/(-m)
    cost = float(np.squeeze(cost))
    #先将[[]]转为float
    assert(isinstance(cost,float))
    
    return cost

#print("---------测试compute_cost----------------")
#A2, Y_assess, parameters = compute_cost_test_case()
#print("cost = " + str(compute_cost(A2, Y_assess ,parameters)))

def backward_propagation(parameters, cache, X, Y):
    """
    参数：
    parameters：神经网络参数的相关字典
    cache：存储正向传播的数据
    X：输入数据，维度(n_x,m)
    Y：输入数据标签，维度(n_y,m)
    返回值：
    grads：包含w和b的字典
    """
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    #根据反向传播计算
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2 }
    
    return grads

#print("---------测试back_forward----------------")
#parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

#grads = backward_propagation(parameters, cache, X_assess, Y_assess)
#print ("dW1 = "+ str(grads["dW1"]))
#print ("db1 = "+ str(grads["db1"]))
#print ("dW2 = "+ str(grads["dW2"]))
#print ("db2 = "+ str(grads["db2"]))

#更新参数
#/theta 参数 /alpha 学习率
#/theta = /theta - （/alpha * d/theta)

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    参数：
    parameters:包含参数的字典
    grads:包含导数值的字典类型的变量。
    learning_rate：学习速率
    返回值：
    parameters:跟新参数后的字典
    """
    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]
    
    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

#print("---------测试update_parameters----------------")
#parameters, grads = update_parameters_test_case()
#parameters = update_parameters(parameters, grads)
#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))
#print("W2 = " + str(parameters["W2"]))
#print("b2 = " + str(parameters["b2"]))

#创建神经网络模型
def nn_model(X, Y, n_h, num_iterations, print_cost = False):
    """
    参数：
    X：数据集合
    Y：标签集合
    n_h：隐藏层数量
    num_iterations：梯度下降循环中的迭代次数
    print_cost：如果为True，则每1000次迭代打印一次成本数值
    返回值：
    parammeters： 模型学习的参数，它们可以用来进行预测
    """
    np.random.seed(3) #指定随机种子
    
    #求输入层、输出层
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]
    
    #初始化
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters,grads,learning_rate = 0.5)
        
        if print_cost:
            #每迭代1000次打印一次结果
            if i % 1000 == 0:
                print("第 ",i," 次循环，成本为："+str(cost))
    return parameters


#print("---------测试nn_model----------------")
#X_assess, Y_assess = nn_model_test_case()

#parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))
#print("W2 = " + str(parameters["W2"]))
#print("b2 = " + str(parameters["b2"]))

def predict(parameters, X):
    """
    参数：
    parameters:包含参数的字典
    X：输入数据，维度数(n_x, m)
    返回值：
    predictions：模型预测的向量（0红色，1蓝色）
    """
    #利用更新后的参数进行前向传播
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    
    return predictions

#print("---------测试predict----------------")
#parameters, X_assess = predict_test_case()

#predictions = predict(parameters, X_assess)
#print("预测的平均值 = " + str(np.mean(predictions)))

print("---------正式测试----------------")
print("---------迭代次数----------------")
parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)

#绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print ('准确率: %d' % float((np.sum(Y == predictions)) / float(Y.shape[1]) * 100) + '%')

#隐藏层个数影响
print("---------隐藏层个数----------------")
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] #隐藏层数量
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.sum(Y == predictions)) / float(Y.size) * 100)
    print ("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))


# In[ ]:




