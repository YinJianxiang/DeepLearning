#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import matplotlib.pyplot as plt
import h5py

from lr_utils import load_dataset

# numpy是使用Python进行科学计算的基本软件包。
# h5py是与存储在H5文件中的数据集进行交互的通用软件包。
# matplotlib是著名的Python图形库。
# PIL和scipy来测试带有您自己的图片模型，

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

# train_set_x_orig ：保存的是训练集里面的图像数据（本训练集有209张64x64的图像）。
# train_set_y_orig ：保存的是训练集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
# test_set_x_orig ：保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
#t est_set_y_orig ： 保存的是测试集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
# classes ： 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。


#index = 5
#imshow用来绘制图像，第一个参数是 绘制的图选或数组
#M*N此时数组必须为浮点数，颜色为灰度；M*N*3数据为浮点数或unit8（0~255），颜色为RGB
#plt.imshow(train_set_x_orig[index])

#打印出当前训练的标签值

print("y = " + str(train_set_y[:,index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture")

# m_train 训练集图片数
# m_test 测试集图片数 
# num_px 图像高度or宽度
# train_set_x_orig 是一个维度为(m_train，num_px，num_px，3）的数组

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

print ("训练集的数量: m_train = " + str(m_train))
print ("测试集的数量 : m_test = " + str(m_test))
print ("每张图片的宽/高 : num_px = " + str(num_px))
print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
print ("测试集_标签的维数: " + str(test_set_y.shape))

#把（64，64，3）的numpy数组构造成（64 x64 x3，1），每张图片由64x64像素构成的，而每个像素点由（R，G，B）三原色构成的。
#1.X_flattrn = X.reshape(X.shape[0],-1)
#将（a,b,c,d）->（a,b*c*d） 这里规定了数组变为209行，列数直接用-1代替就行了，计算机帮我们算有多少列

#2.转置之后，（b*c*d,a）

train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print ("测试集_标签的维数 : " + str(test_set_y.shape))

#预处理，进行数据标准化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

#预处理步骤
#1.找出问题的尺寸和形状
#2.重塑数据集
#3 .标准化数据

#建立神经网络步骤
#1.定义模型结构（输入特征的数量）
#2.初始化模型的参数
#3.循环
# 计算当前损失（正向传播）
# 计算当前梯度（反向传播）
# 更新参数（梯度下降）

#构建sigmoid（）函数

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

#初始化参数
# logstic函数初始化可以
#

def initialize_with_zeros(dim):

    """
    为w创建一个(dim,0)的0向量，b初始化为0
    参数：
    dim： w向量的大小（dim，1）
    返回：
    w： （dim,1）的初始化的向量
    b: 初始化的标量
    """
    w = np.zeros((dim,1))
    b = 0
#assert()如果它的条件返回错误，则终止程序执行
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    
    return (w,b)
    
#分别进行正向传播和反向传播

def propagate(w,b,X,Y):
    
    """
    参数：
    w 权重，(num_px * num_px * 3,1)
    b 偏差
    X 训练数据，（num_px * num_px * 3,训练数量）
    Y 标签（非猫为0，猫为1），(1,训练数量)
    
    返回：
    cost 逻辑回归的负对数似然成本
    dw 相对于w的损失程度，
    db 相对于b的损失程度，
    
    """
    
    #训练数量
    m = X.shape[1]
    
    #正向传播
    A = sigmoid(np.dot(w.T,X) + b)
    cost = -1 / m  * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
    #反向传播
    dw = (1 / m) * np.dot(X,(A - Y).T)
    db = (1 / m) * np.sum(A - Y) 
    
    # numpy.squeeze(a,axis = None)
    # a为输入数组
    # axis用于删除指定维度，但指定维度必须为单维度，默认则删除所有单维度
    # 返回值：数组
    # 不会修改原数组的值
    
    #使用断言确保数据正确
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    #创建一个字典，存储dw和db
    grads = {
        "dw":dw,
        "db":db
        
    }
    
    return (grads,cost)


#测试propagate
w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[1,2]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)

print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))
print("grads = " + str(grads))

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = True):
    
    """
    参数
    w 权重，(num_px * num_px * 3,1)
    b 偏差
    X 训练数据，（num_px * num_px * 3,训练数量）
    Y 标签（非猫为0，猫为1），(1,训练数量)
    num_iteration 优化循环的迭代次数
    learning_rate 梯度下滑更新规则的学习率
    print_cost 每100步
    
    返回
    param 包含权重w和偏差b的字典
    grads 包含权重和偏差梯度的字典
    costs 优化期间的成本列表
    
    步骤：
    1.计算当前的成本和梯度，使用propagate()
    2.使用w和b的梯度下降法更新参数
    
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        #计算成本和梯度
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        #记录成本
        if i % 100 == 0:
            costs.append(cost)
        #打印
        if (print_cost) and (i % 100 == 0):
             print("迭代的次数: %i ， 误差值： %f" % (i,cost))
            
    
    params = {
        "w": w,
        "b": b
    }
    
    grads = {
        "dw": dw,
        "db": db
    }
    
    return (params, grads, costs)


#测试
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
params , grads , costs = optimize(w , b , X , Y , num_iterations=100 , learning_rate = 0.009 , print_cost = True)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

#利用已经学习的w和b，对于测试集进行预测

def predict(w, b, X):
    """
    参数：
    w 权重，(num_px * num_px * 3,1)
    b 偏差
    X 测试集数据，(num_px * num_px * 3,测试数据数量)

    返回值：
    Y_prediction 标签向量（0|1）
    """

    #测试集数量
    m = X.shape[1]
    
    Y_prediction = np.zeros((1,m))
    #为了让权重w和测试集特征对应
    w = w.reshape(X.shape[0],1)
    
    A = sigmoid(np.dot(w.T,X) + b)
        
    for i in range(m):
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,1] = 0
            
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction

#predict()测试
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print("predictions = " + str(predict(w, b, X)))

#进行功能整合

def model(X_train , Y_train , X_test , Y_test , num_iterations = 2000 , learning_rate = 0.5 , print_cost = False):
    """
    参数：
        X_train 训练集，（num_px * num_px * 3，m_train）
        Y_train 训练集对应的标签集合，（1，m_train）
        X_test  测试集，（num_px * num_px * 3，m_test）
        Y_test  测试集对应的标签集合，（1，m_test）
        num_iterations  优化过程的迭代次数
        learning_rate  梯度下滑时的学习率
        print_cost  每100步打印成本
    
    返回：
        d  包含有关信息的字典。
    """
    
    #初始化w和b
    w , b = initialize_with_zeros(X_train.shape[0])
    
    #parameterx(w,b) grads(dw,db)
    parameters , grads , costs = optimize(w , b , X_train , Y_train,num_iterations , learning_rate , print_cost)
    
    w , b = parameters["w"] , parameters["b"]
    
    #预测测试集
    Y_prediction_test = predict(w , b, X_test)
    Y_prediction_train = predict(w , b, X_train)
    
    #打印训练后的准确性
    print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) ,"%")
    print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) ,"%")
    
    d = {
            "costs" : costs,
            "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations }
    return d

#测试model
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#index = 5
#把第index列 转化为 64*64*3
#plt.imshow(test_set_x[:,index].reshape(num_px, num_px, 3))

#绘制图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

learning_rates = [0.005, 0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    print("num_iterations=2000")
    models[str(i)] = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations = 2000, learning_rate = i, print_cost = True)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# In[ ]:




