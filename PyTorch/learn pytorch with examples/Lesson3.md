### numpy

在了解简单神经网络的具体结构，就可以用numpt来实现简单的网络。但是需要自己实现正向和反向传播

```python
# -*- coding: utf-8 -*-
import numpy as np
import math

#使用numpy实现神经网络

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
```

### with Pytorch

numpy不能使用gpu进行加速，不适合进行深度神经网络的搭建。

tensor类似numpy，是一种N维数组，能构建计算图和梯度，tensor能构建计算图和梯度，为后面的自动微分autograd做准备。

同时Pytorch能使用GPU进行加速。

```python
# -*- coding: utf-8 -*-

import torch
import math

#PyTorch Tensor 在概念上与 numpy 数组相同：Tensor 是一个 n 维数组，PyTorch 提供了许多操作这些 Tensor 的函数。
#PyTorch Tensors 可以利用 GPU 来加速其数值计算。要在 GPU 上运行 PyTorch Tensor，您只需指定正确的设备。

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

### Tensors and autograd

PyTorch提供了自动微分机制，来自动化神经网络反向传播的计算。

当使用autograd，前向传播网络需要定义一个计算图，图中的节点就是张量，边就是输入某一张量、输出另一张量的操作。反向传播可以通过这个计算图非常方便地计算梯度。

计算图听起来很复杂，但在实践中其实很简单。每个张量都代表计算图中的一个节点，如果`x`是一个张量且`x.requires_grad=True`，那么`x.grad`就是另一个张量，用来存储`x`的梯度信息。

```python
# -*- coding: utf-8 -*-
import torch
import math

#使用PyTorch中的 autograd包自动计算反向传播梯度

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device = device, dtype = dtype)
y = torch.sin(x)

#y = a + b x + c x^2 + d x^3
a = torch.randn((), device = device, dtype = dtype, requires_grad = True)
b = torch.randn((), device = device, dtype = dtype, requires_grad = True)
c = torch.randn((), device = device, dtype = dtype, requires_grad = True)
d = torch.randn((), device = device, dtype = dtype, requires_grad = True)

learning_rate = 1e-6

for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    
    loss = (y_pred - y).pow(2).sum()
    
    if t % 100 == 99:
        print(t, loss.item())
    #使用autograd来计算向后传递
    #这个调用将计算关于requires_grad=True的所有张量的损失梯度
    loss.backward()
    
    with torch.no_grad():
    #torch.no_grad()  不跟踪梯度变化
    #由于在更新权重时候不需要跟踪梯度变化    
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        # 更新权重后，手动将梯度置零,否则会叠加之前的值
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

### Defining new autograd function

一般autograd操作包括：

1. 前向传播
2. 后向传播

在PyTorch中我们也可以定义自己的autograd操作，继承`torch.autograd.Function`，并且实现`forward`和`backward`即可。

```python
# -*- coding: utf-8 -*-
import torch
import math

# 通过定义torch.autograd.Function和实现forward和backward函数的子类来轻松定义 autograd 运算符。

class LegendrePolynomial3(torch.autograd.Function):
    """
    ctx是context的缩写，上下文，环境
    ctx专门用在静态方法中，调用不需要实例化对象，直接通过类名就可以调用
    自定义的forward()方法和backward()方法的第一个参数必须是ctx; ctx可以保存forward()中的变量,以便在backward()中继续使用
    ctx.save_for_backward(a, b)能够保存forward()静态方法中的张量, 从而可以在backward()静态方法中调用, 具体地, 通过a, b = ctx.saved_tensors重新得到a和b
    ctx.needs_input_grad是一个元组, 元素是True或者False, 表示forward()中对应的输入是否需要求导
    """
    @staticmethod
    def forward(ctx, input): 
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grag_output):
        input, = ctx.saved_tensors
        return grag_output * 1.5 * (5 * input ** 2 - 1)
        
dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device = device, dtype = dtype)
y = torch.sin(x)

a = torch.full((), 0.0, device = device, dtype = dtype,requires_grad = True)
b = torch.full((), -1.0, device = device, dtype = dtype,requires_grad = True)
c = torch.full((), 0.0, device = device, dtype = dtype,requires_grad = True)
d = torch.full((), 0.3, device = device, dtype = dtype,requires_grad = True)

learning_rate = 5e-6

for t in range(2000):
    #apply Fuction
    P3 = LegendrePolynomial3.apply
    
    y_pred = a + b * P3(c + d * x)
    loss = (y_pred - y).pow(2).sum()
    
    if t % 100 == 99:
        print(t, loss.item())
    
    loss.backward()
    
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
    
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')
```

Pytorch采用动态计算图，tensorflow采用静态计算图

静态计算图: 只对计算图定义一次, 而后会多次执行这个计算图.
好处:

> 可以预先对计算图进行优化, 融合一些计算图上的操作, 并且方便在分布式多GPU或多机的训练中优化模型

动态计算图: 每执行一次都会重新定义一张计算图.

> 控制流就像Python一样, 更容易被人接受, 可以方便的使用for, if等语句来动态的定义计算图, 并且调试起来较为方便.

## nn Moudle

nn包定义了一系列Modules，类似于神经网络的各个layers。一个module能够接收input tensors，计算output tensors，也能够存储中间状态（比如learnable parameters）。

nn包还定义了一系列在训练中常使用的loss function

```python
# -*- coding: utf-8 -*-
import torch
import math


#nn包定义了一组Modules，大致相当于神经网络层。模块接收输入张量并计算输出张量，但也可以保存内部状态，例如包含可学习参数的张量。
#nn包还定义了一组在训练神经网络时常用的有用的损失函数

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
#y = x.unsqueeze(-1)
#print(y.shape)

#广播 (2000,1)  (3，) -> (2000,3)
xx = x.unsqueeze(-1).pow(p)


#nn.Sequential
#一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
#同时以神经网络模块为元素的有序字典也可以作为传入参数。
#Linear y = xAT + b使用线性函数计算输入的输出，并保存其权重和偏差的内部张量
#Linear(in_features, out_features, bias = True)
#输入特征数，输出特征数
#Flatten(x,y) 从x维到y维推平，保证输出层1维tensor，匹配y
model = torch.nn.Sequential (
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

#m = torch.nn.Linear(3, 1)
#print(m.weight.shape)
#output = m(xx)
#print(output.shape)
#Linear生成(1，3)矩阵，在转置为(3,1)

#选择MSE计算损失函数 y = (1/m) sum(y - y')^2
#reduction sum or mean
loss_fn = torch.nn.MSELoss(reduction = 'sum')

learning_rate = 1e-6


#preds = model(inputs)             ## inference
#loss = criterion(preds, targets)  ## 求解loss
#optimizer.zero_grad()             ## 梯度清零
#loss.backward()                   ## 反向传播求解梯度
#optimizer.step()                  ## 更新权重参数


for t in range(2000):
    y_pred = model(xx)
    
    loss = loss_fn(y_pred, y)
    
    if t % 100 == 99:
        print(t, loss.item())

    #梯度清0
    model.zero_grad()
    
    #反向传播
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

linear_layer = model[0]

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
```

### optim

optim包提供了各种优化算法，如Momentun、RMSProp、Adam等算法

```python
import torch
import math

#optim包中有优化函数算法，并提供了常用优化算法的实现。

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3,1),
    torch.nn.Flatten(0,1)
)

loss_fn = torch.nn.MSELoss(reduction = 'sum')

learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)

for t in range(2000):
    y_pred = model(xx)
    
    loss = loss_fn(y_pred, y)
    
    if t % 100 == 99:
        print(t, loss.item())
        
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()

linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

```

### Custom nn Module

可以直接继承`torch.nn.Module`来定义自己的Module，实现forward方法就可以。

反向传播依赖构建的计算图，不需要手动计算。

```python
# -*- coding: utf-8 -*-
import torch
import math

#自定义nn模型
#我们需要更复杂的模块， 我们可以通过 继承 nn.Module  
#和定义 forward（用来接收Input tensor和输出output tensor）

class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        实例化四个参数，并指定为成员参数
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        
    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
    
    def string(self):
                return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'
        

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)
model = Polynomial3()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr= 1e-6)

for t in range(2000):
    y_pred = model(x)
    loss= criterion(y_pred,y)
    if t % 100 == 99:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')
```

### Control Flow + Weight Sharing

作为动态图和权重共享的一个例子，我们实现了一个非常奇怪的模型：一个三到五阶的多项式，它在每次向前传递时选择一个3到5之间的随机数，并使用那么多的阶数，多次重复使用相同的权重来计算第四阶和第五阶。

对于这个模型，我们可以使用普通的Python流控制来实现循环，并且我们可以通过在定义前向传递时多次重用相同的参数来实现权重共享。

```python
import torch
import random
import math

#动态图和权重共享

class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))
    
    def forward(self, x):
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x **3
        for exp in range(4, random.randint(4, 6)):
            y += self.e * x ** exp
        return y

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'
    
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = DynamicNet()

criterion = torch.nn.MSELoss(reduction='sum')
#class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-8, momentum = 0.9)

for t in range(30000):
    y_pred = model(x)
    
    loss = criterion(y_pred, y)

    if t % 2000 == 1999:
        print(t, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')
```

