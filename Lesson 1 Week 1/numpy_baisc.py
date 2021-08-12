#!/usr/bin/env python
# coding: utf-8

# In[2]:


test = "Hello world!"
print("test:"+ test)


# In[5]:


#import math

#def basic_sigmoid(x):
#    s = 1 / (1+math.exp(-x))
#    return s

#a = basic_sigmoid(3)
#print(a)


# In[9]:


import numpy as np

x = np.array([1,2,3])

print(x+1)
print(np.exp(x))


# In[15]:


import numpy as np

def basic_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

y = np.array([1,2,3])

z = basic_sigmoid(y)

print(z)


# In[19]:


import math

def sigmoid_derivative(x):
    s = basic_sigmoid(x)
    ds = s * (1-s)
    return ds

x = np.array([1,2,3])
sigmoid_derivative(x)


# In[22]:


#假定image为m*n*q维
import numpy

def image2vector(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2],1)
    return v

image = numpy.array([[[1,2],
    [3,4],
    [5,6]],
    [[7,8],
     [9,10],   
     [11,12]],
    [[13,14],
     [15,16],   
     [17,18]]])

image2vector(image)


# In[27]:


import numpy as py
# norm(x, ord = None, axis = None, keepdims = False)
# 默认
# ord = 2 sqrt(sum(xi^{2}))
# ord = 1 sum(|xi|)
# ord = np.inf max(|xi|)

def normalizeRows(x):
    x_norm = np.linalg.norm(x,axis = 1,keepdims = True)
    x = x / x_norm
    return x

x = np.array([[0,3,4],
              [2,6,4]  
    ])

normalizeRows(x)


# In[29]:


import numpy as np

def softmax(x):
    x_exp = np.exp(x);
    x_sum = np.sum(x_exp,axis = 1,keepdims = 1)
    s = x_exp / x_sum
    return s

x = np.array([
        [9,2,5,0,0],
        [7,5,0,0,0]
])

softmax(x)


# In[34]:


import numpy as np
x1 = [9,2,5,0,0,7,5,0,0,0,9,2,5,0,0]
x2 = [9,2,2,9,0,0,2,5,0,0,9,2,5,0,0]

np.dot(x1,x2)

#求向量点积 a dot b = sum(aibi)
x3 = np.array([
        [2,2,1],
        [3,2,1]
])

x4 = np.array([
        [3],
        [1],
        [5]
])

#np.dot(x3,x4)

#外积 先把a，b转化为一维向量 形成m*n数组
#a:m*1 b:1*n 

np.outer(x1,x2)

#np.outer(x3,x4)

#multiply 对应每个元素相乘


# In[ ]:





# In[ ]:




