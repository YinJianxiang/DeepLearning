### Pyporch编写代码的主要步骤

主要分为四大步骤

1. 输入数据处理(X 输入数据，数据转换为Tensor)
2. 模型构建模块(前向传播，从输入数据得到预测$\hat{y}$)
3. 定义代价函数和优化器模块(eg:前向过程只会得到模型预测的结果，并不会自动求导和更新，是由这个模块进行处理)
4. 构建训练过程(迭代整个训练过程)

#### 数据处理

主要需要进行数据转换为Tensor。pyporch提供了Dataset和Dataloader（mini-batch）两个类方便构建

```python
torch.utils.data.Dataset
torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)
```

#### 模型构建

所有的模型都需要继承$torch.nn.Module$，另外需要实现$forward()$方法，$backward()$模型自动实现

#### 代价函数和优化器

构造损失函数(loss function)和优化器(Optimizer)

```python
criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.9)
```

#### 构建训练过程

```python
def train(epoch):
	for i, data in emumerate(dataloader, 0):
        x, y = data  # 取出minibatch数据和标签
        y_pred = model(x)  # 前向传播
        loss = criterion(y_pred, y)  # 计算代价函数
        optimizer.zero_grad()  # 清零梯度准备计算
        1oss.backward()  # 反向传播
        optimizer.step()  # 更新训练参数
```

