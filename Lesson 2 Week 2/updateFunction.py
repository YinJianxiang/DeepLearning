#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np 
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import opt_utils
import testCase

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    功能：
    使用梯度下降更新参数
    
    参数：
    parameter:包含参数的字典
    parameters['W' + str(l)] = Wl
    parameters['b' + str(l)] = bl
   
   grads:包含梯度值的字典
    grads['dW' + str(l)] = dWl
    grads['db' + str(l)] = dbl
    
    learning_rate:学习率
    返回值：
    parameters:包含参数的字典
    """
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters

#测试update_parameters_with_gd
print("-------------测试update_parameters_with_gd-------------")
parameters , grads , learning_rate = testCase.update_parameters_with_gd_test_case()
parameters = update_parameters_with_gd(parameters,grads,learning_rate)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

#mini-batch梯度下降
#第一个mini-batch
#first_mini_batch_X = shuffled_X[:, 0 : mini_batch_size]
#第二个mini-batch
#second_mini_batch_X = shuffled_X[:, mini_batch_size : 2 * mini_batch_size]

def random_mini_batches(X, Y, mini_batch_size, seed = 0):
    """
    功能：
    从(X,Y)中创建一个随机的mini-batch列表
    参数：
    X:输入数据，维度(n_x,m)
    Y:输入数据的标签，维度(1,m)
    mini_batch_size:mini_batch的样本数量
    seed:随机种子
    返回值：
    mini_batches:mini_batch序列
    """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    #打乱顺序并对应匹配
    permutation = list(np.random.permutation(m))
    shuffled_X = X[: ,permutation]   
    shuffled_Y = Y[: ,permutation].reshape((1, m))
    
    #分割
    num_complete_minibatches = math.floor(m / mini_batch_size)
    #扫地除法
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k+1) * mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    #处理剩余的数据
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:,mini_batch_size * num_complete_minibatches:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches

#测试random_mini_batches
print("-------------测试random_mini_batches-------------")
X_assess,Y_assess,mini_batch_size = testCase.random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess,Y_assess,mini_batch_size)

print("第1个mini_batch_X 的维度为：",mini_batches[0][0].shape)
print("第1个mini_batch_Y 的维度为：",mini_batches[0][1].shape)
print("第2个mini_batch_X 的维度为：",mini_batches[1][0].shape)
print("第2个mini_batch_Y 的维度为：",mini_batches[1][1].shape)
print("第3个mini_batch_X 的维度为：",mini_batches[2][0].shape)
print("第3个mini_batch_Y 的维度为：",mini_batches[2][1].shape)

#包含动量的梯度下降
#v_dW[l] = \beta * v_dW[l] + (1 - \beta)dW[l] 
#W[l] = W[l] - learning_rate * v_dw[l]

#v_db[l] = \beta * v_db[l] + (1 - \beta)db[l] 
#b[l] = b[l] - learning_rate * v_db[l]


def initialize_velocity(parameters):
    """
    功能:
    初始化速度v字典
    keys:"dW1", "db1", ..., "dWL", "dbL"
    values:对应的速度
    参数:
    parameters:包含参数的字典
    返回值:
    v:包含速度的字典
    
    """
    
    L = len(parameters) // 2
    v = {}
    
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
        
    return v

#测试initialize_velocity
print("-------------测试initialize_velocity-------------")
parameters = testCase.initialize_velocity_test_case()
v = initialize_velocity(parameters)

print('v["dW1"] = ' + str(v["dW1"]))
print('v["db1"] = ' + str(v["db1"]))
print('v["dW2"] = ' + str(v["dW2"]))
print('v["db2"] = ' + str(v["db2"]))


def update_parameters_with_momentun(parameters,grads,v,beta,learning_rate):
    """
    功能:
    包含动量的梯度下降
    参数:
    parameter:包含参数的字典
    grads:包含梯度的字典
    v:包含速度的字典
    beta:超参数，动量
    learning_rate:学习率
    返回值:
    parameter:更新后的参数字典
    v:更新后的速度字典
    """

    L = len(parameters) // 2
    
    for l in range(L):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
    
    return parameters,v    

#测试update_parameters_with_momentun
print("-------------测试update_parameters_with_momentun-------------")
parameters,grads,v = testCase.update_parameters_with_momentum_test_case()
update_parameters_with_momentun(parameters,grads,v,beta = 0.9,learning_rate = 0.01)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print('v["dW1"] = ' + str(v["dW1"]))
print('v["db1"] = ' + str(v["db1"]))
print('v["dW2"] = ' + str(v["dW2"]))
print('v["db2"] = ' + str(v["db2"]))


#adam算法

def initialize_adam(parameters):
    """
    功能:
    初始化v和s
    参数:
    parameter:包含参数的字典
    返回值:
    v:包含梯度的指数加权平均值的字典
    s:包含平方梯度的指数加权平均值的字典
    """
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v,s

#测试initialize_adam
print("-------------测试initialize_adam-------------")
parameters = testCase.initialize_adam_test_case()
v,s = initialize_adam(parameters)

print('v["dW1"] = ' + str(v["dW1"])) 
print('v["db1"] = ' + str(v["db1"])) 
print('v["dW2"] = ' + str(v["dW2"])) 
print('v["db2"] = ' + str(v["db2"])) 
print('s["dW1"] = ' + str(s["dW1"])) 
print('s["db1"] = ' + str(s["db1"])) 
print('s["dW2"] = ' + str(s["dW2"])) 
print('s["db2"] = ' + str(s["db2"])) 

def update_parameters_with_adam(parameters, grads, v, s, t , learning_rate = 0.01, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8):
    """
    功能:
    使用Adam更新参数
    参数:
    parameter:包含参数的字典
    grads:包含梯度的字典
    v:包含梯度的指数加权平均值的字典
    s:包含平方梯度的指数加权平均值的字典
    t:当前的迭代次数
    learing_rate:学习率
    beta1:动量，超参数
    beta2:RMSprop的一个参数，超参数
    epsilon:防止分母为0的参数，一般为1^-8
    返回值:
    parameter:更新后的参数字典
    v:更新后的包含梯度的指数加权平均值的字典
    s:更新后的包含平方梯度的指数加权平均值的字典
    """
    L = len(parameters) // 2
    #偏差修正后的值
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]
        
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
        
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])
        
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
        
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (v_corrected["dW" + str(l + 1)] / (np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (v_corrected["db" + str(l + 1)] / (np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)))
    
    
    return parameters, v, s 

#测试update_with_parameters_with_adam
print("-------------测试update_with_parameters_with_adam-------------")
parameters , grads , v , s = testCase.update_parameters_with_adam_test_case()
update_parameters_with_adam(parameters,grads,v,s,t=2)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print('v["dW1"] = ' + str(v["dW1"])) 
print('v["db1"] = ' + str(v["db1"])) 
print('v["dW2"] = ' + str(v["dW2"])) 
print('v["db2"] = ' + str(v["db2"])) 
print('s["dW1"] = ' + str(s["dW1"])) 
print('s["db1"] = ' + str(s["db1"])) 
print('s["dW2"] = ' + str(s["dW2"])) 
print('s["db2"] = ' + str(s["db2"])) 


# In[ ]:




