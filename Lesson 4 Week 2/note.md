## keras框架

类似tensorflow和pytorch，是一种更高级的框架

```python
batch_normalization()
```

axis对应channel



### 测试模型

1. 创建模型

2. 编译模型

   ```python
   model.compile(optimizer = "", loss = "", metrics = ["accuracy"])
   ```

3. 训练模型

   ```python
   model.fit(x = , y = , batch_size = )
   ```

4. 评估模型

   ```python
   model.evaluate(x = , y = )
   ```



> 在 fit 和 evaluate 中 都有 verbose 这个参数
>
> fit 中的 verbose
>
> verbose：该参数的值控制日志显示的方式
> verbose = 0    不在标准输出流输出日志信息
> verbose = 1    输出进度条记录
> verbose = 2    每个epoch输出一行记录
> 注意： 默认为 1

### model.evaluate

输入数据和标签,输出损失和精确度.

	# 评估模型,不输出预测结果
	loss,accuracy = model.evaluate(X_test,Y_test)
	print('\ntest loss',loss)
	print('accuracy',accuracy)

### model.predict

输入测试数据,输出预测结果
(通常用在需要得到预测结果的时候)

    #模型预测,输入测试集,输出预测结果
    ![残差网络](D:\DeepLearning\Lesson 4 Week 2\残差网络.png)y_pred = model.predict(X_test,batch_size = 1)

两者差异

1输入输出不同
model.evaluate输入数据(data)和金标准(label),然后将预测结果与金标准相比较,得到两者误差并输出.
model.predict输入数据(data),输出预测结果
2是否需要真实标签
model.evaluate需要,因为需要比较预测结果与真实标签的误差
model.predict不需要,只是单纯输出预测结果,全程不需要金标准的参与.

## 残差网络

为了解决深层网络的梯度爆炸or梯度消失

![](D:\DeepLearning\Lesson 4 Week 2\残差网络.png)

### 恒等快

输入激活值$a^{[i]}$和输出激活值$a^{[i+1]}$要有相同的维度，所以需要BatchNorm

![](D:\DeepLearning\Lesson 4 Week 2\跳跃两层.png)



$x+x(shortcut)$，然后再进行激活

valid不填充

假设使用跳跃三层

![](D:\DeepLearning\Lesson 4 Week 2\跳跃三层.png)

第一部分：

1. 第一个卷积层$F_{1}$个滤波器，f = (1,1)，stride = (1, 1)，valid填充
2. 规范层BatchNorm是通道的轴归一化
3. ReLU activate

第二部分：

1. 第二个CONV2D有$F_{2}$个滤波器，f = (f, f)，stride = (1,1)，same填充
2. 规范层BatchNorm是通道的轴归一化
3. ReLU activate

第三部分：

1. 第三个CONV2D有$F_{3}$个滤波器，f = (1, 1)，stride = (1, 1)，valid填充
2. 规范层BatchNorm是通道的轴归一化

`x(shortcut)`：

X



最后

$x+x(shortcut)$，然后再进行激活

### 卷积块

![](D:\DeepLearning\Lesson 4 Week 2\卷积块.png)

第一部分：

1. 第一个卷积层$F_{1}$个滤波器，f = (1,1)，stride = (s, s)，valid填充
2. 规范层BatchNorm是通道的轴归一化
3. ReLU activate

第二部分：

1. 第二个CONV2D有$F_{2}$个滤波器，f = (f, f)，stride = (1,1)，same填充
2. 规范层BatchNorm是通道的轴归一化
3. ReLU activate

第三部分：

1. 第三个CONV2D有$F_{3}$个滤波器，f = (1, 1)，stride = (s, s)，valid填充
2. 规范层BatchNorm是通道的轴归一化

`x(shortcut)`：

1. 卷积层有$F_{3}$个过滤器，f = (1,1)，stride = (s,s)，valid填充
2. 规范层BatchNorm是通道的轴归一化

最后

$x+x(shortcut)$，然后再进行激活



## 50层残差网络

流程图

![](D:\DeepLearning\Lesson 4 Week 2\50层残差网络.png)

`ID BLOCK`代表恒等块，`CONV BLOCK`代表卷积块

具体流程：

1. 0填充，`padding=(3,3)`
2. stage 1:
   1. 卷积层64个过滤器，f = (7,7)，stride = (2,2)，命名conv1
   2. 规范层BatchNorm是通道的轴归一化
   3. ReLU activate
   4. max pool f = (3,3)，stride = (2,2)
3. stage 2:
   1. 卷积块 filters = [64,64,256]，f=3，s=1，block = "a"
   2. 恒等块 filters = [64,64,256]，f = 3，block = "b" "c"
4. stage 3:
   1. 卷积块 filters = [128,128,512]，f=3，s=2，block = "a"
   2. 3个恒等块  filters = [128,128,512]，f=3，block = "b" "c" "d"
5. stage 4:
   1. 卷积块  filters = [256,256,1024]，f=3，s=2，block = "a"
   2. 5个恒等块 filters = [256,256,1024]，f=3，block = "b" "c" "d" "e" "f"
6. stage 5:
   1. 卷积块 filters = [512,512,2056]，f=3，s=2，block = "a"
   2. 2个恒等块 filters = [256,256,2048]，f = 3，block = "b" "c"
7. average pool f = (2,2)
8. Flatten
9. Full connect 用softmax

