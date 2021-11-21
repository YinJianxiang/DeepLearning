### What is PyTorch

#### Pytorch是什么

- numpy的GPU版实现
- 深度学习研究平台

#### 创建tensor

```python
x = torch.empty(5, 3)
x = torch.rand(5, 3)
x = torch.zeros(5, 3, dtype = torch.long)
x = torch.tensor([5.5, 3])
x.new_ones(5, 3, dtype = torch.double)
torch.randn_like(x, dtype = torch.float)
```

#### Tensor的基本操作

*x.size()*

获得Tensor尺寸,返回值torch.Size对象

- torch.Size([5,3])
- 数据类型为tuple

*向量加法*

```python
x.size()
x + y
z = torchch.add(x, y)
torch.add(x, y, result)
y.add_(x)
#相当于y += x
```

*reshape()操作*

```python
x.view(-1,8)
```

*获取元素*

```python
x.item()
```

#### tensor和numpy的相互转化

- tensor->numpy

```python
x.numpy()
```

- numpy->tensor

```python
torch.from_numpy(ndarray)
```

#### CUDA tensors

- 准备工作 `device = torch.device("cuda")`
- 直接定义CUDA tensor ‘`torch.ones_like(x, device=device)`’
- CPU Tensor转换为CUDA tensor `cpu_tensor.to(device)`
- CUDA Tensor转换为CPU tensor `cuda_tensor.to("cpu", torch.double)`

### Autograd: Automatic differentiation

`autograd`为张量的所有操作提供了自动求导机制。使用运行时定义(define-by-run)的框架，反向传播的将根据代码运行决定，每次迭代都不一样

`torch.Tensor`的`.require_grad`为True，将会追踪对于该张量的所有操作。当完成计算后可以通过调用 `.backward()`，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到`.grad`属性

```python
x = torch.ones(2, 2, requires_grad = True)
```

要阻止张量被跟踪历史，可以调用`.detach()` 方法将其与计算历史分离，并阻止它未来的计算记录被跟踪

被`with_torch.no_grad()`包括的代码，不用跟踪反向梯度计算

Tensor和Function互相连接并构建一个计算图，它编码完整的计算过程。每个张量都有一个`.grad_fn`属性引用Function已创建的属性Tensor

如果tensor是一个标量，就可以省略`gradient`参数，而如果这个tensor是向量，则必须指明`gradient`参数

```python
x = torch.tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad = True)
print(x)

y = x + 2
print(y)

z = y * y * 3
print(z)

out = z.mean()
print(out)

out.backward()
print(x.grad)

print(y.grad)
print(z.grad)
print(out.grad)
```

![1634401717719](C:\Users\YinJianxiang\AppData\Roaming\Typora\typora-user-images\1634401717719.png)

### Neural Networks

#### Pytorch中搭建神经网络

- 一般基于`torch.nn`类
- `nn`可以通过`autograd`来计算微分，所有模型（各层layer）都继承`nn.Module`，子类的成员函数`forward(input)`返回模型的运行结果

神将网络的过程包含以下过程：

- 定义一个神经网络，包括需要训练的参数
- 遍历Dataset获取训练输入数据
- 通过正向传播得到模型输出结果
- 计算损失函数
- 通过自动微分求导
- 更新参数：`weight = weight - lr * gradient`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        #1 input image channel, 6 output channels, 5x5 square convolution
        #kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        #卷积层，输入1通道灰度图，输出6通带特征图，卷积核大小3*3
        self.conv2 = nn.Conv2d(6, 16, 5)
        #卷积层，输入6通道特征图，输出16通带特征图，卷积核大小5*5
        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #输入为16*6*6的维度，输出为120的全连接层
        self.fc2 = nn.Linear(120, 84)
        #输入为120的维度，输出为84的全连接层
        self.fc3 = nn.Linear(84, 10)
        #输入为84的维度，输出为10的全连接层
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #1.卷积层
        #2.激活函数
        #3.池化层下采样，具体为在2*2的方格中取一个最大值
        
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        #进行第一层全连接并使用激活函数映射
        
        x = F.relu(self.fc2(x))
        #进行第二层全连接并使用激活函数映射
        
        x = self.fc3(x)
        #第三层直接输出
        
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

#获取模型中的可学习参数
params = list(net.parameters())
#net.parameters()函数来获取模型中可学习的参数
print(len(params))#具有的可学习参数的层数
print(params[0].size()) #conv1的参数

#进行前向传播
input = torch.randn(1,1,32,32)# 四个维度分别为 (N,C,H,W)
out = net(input)#自动调用forward函数计算并返回结果
print(out)


#清空梯度缓存并计算所有需要求导的参数的梯度
net.zero_grad()
out.backward(torch.randn(1,10))

#计算损失函数，一般接收一对数据(output, target),计算两者之间的距离
output = net(input)
target = torch.randn(10)
target = target.view(1,-1) # 令target和output的shape相同.
criterion = nn.MSELoss()#选择计算损失函数方法
loss = criterion(output, target)
print(loss)

#用.grad_fn属性, 可以看到关于loss的计算图
print(loss.grad_fn)
# 当调用loss.backward()时, 就会计算出所有(requires_grad=True的)参数关于loss的梯度, 并且这些参数都将具有.grad属性来获得计算好的梯度

#手动更新参数
learning_rate = 0.001
for f in net.parameters():
    f.data.sub_(learning_rate*f.grad.data)

#利用优化器自动更新梯度
#SGD、Adam算法等
optimizer = optim.SGD(net.parameters(), lr=0.01) # 创建优化器
optimizer.zero_grad() # 清空缓存
output = net(input)
loss = criterion(output, target)
loss.backward() # 计算梯度
optimizer.step() # 执行一次更新
   
```

### Training An Image classifier

主要步骤：

- 使用加载并归一化`CIFAR10`的训练数据集和测试数据集
- 定义一个卷积神经网络
- 定义`Loss function`
- 在`training data`上训练网络
- 在`test data`上测试网络

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

#torchvision的输出类型是 PILImage，转换为Tensor并进行归一化
transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


# 类别信息基本情况
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#查看训练图片样本
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

if __name__ == '__main__':
    #得到一些随机训练图片
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    #展示图片
    imshow(torchvision.utils.make_grid(images))
    #打印标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

训练模型

```python
#定义卷积神经网络
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
net = Net()

#定义损失函数
#损失函数使用交叉熵，优化器使用带动量的SGD
#import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

#训练模型
for epoch in range(2):
    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        input, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("train Finished!")
```

利用模型进行预测

```python
#保存模型
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#测试模型
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))

# 获取少量图片的预测结果
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# 在整个验证集上验证模型结果
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# 查看每一类的预测结果
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

```

在GPU上运行

```python
# GPU上训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
inputs, labels = data[0].to(device), data[1].to(device)
```

