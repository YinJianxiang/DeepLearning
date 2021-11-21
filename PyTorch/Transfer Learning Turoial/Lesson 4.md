通常情况下，我们不会从头训练整个神经网络，更常用的做法是先让模型在一个大数据集进行预训练，然后将预训练模型的权重作为当前任务的初始化参数，或者作为固定的特征提取器来使用。

- Finetuning the convnet: 在一个已经训练好的模型上面进行二次训练
- ConvNet as fixed feature extractor: 此时, 我们会将整个网络模型的权重参数固定, 并且将最后一层全连接层替换为我们希望的网络层. 此时, 相当于是将前面的整个网络当做是一个特征提取器使用.
  

### Load Data

我们将会使用`torch.utils.data`包来载入数据。我们接下来需要解决的问题是训练一个模型来分类蚂蚁和蜜蜂，我们总共拥有120张训练图片, 具有75张验证图片。



### Visualize a few images

可视化图像



### Training the model

现在，编写一个通用函数来训练模型。

- 调整学习比率
- 保存最佳模型

利用LR scheduler对象`torch.optim.lr_scheduler`设置lr scheduler, 并且保存最好的模型。



### Visualizing the model prediction

可视化模型预测，泛型函数，显示对一些图像的预测



### FineTuning the convnet

加载预训练模型, 并重置最后一层全连接层



### Convnet as Fixed Feature Extractor

这里我们需要将除了最后一层的所有网络冻结。我们需要设置`requires_grad == False`去冻结参数以便梯度在backward()中不会被计算

