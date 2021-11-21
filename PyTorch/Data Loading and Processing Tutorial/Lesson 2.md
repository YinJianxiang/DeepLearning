### introduction

对于一个新的机器/深度学习，大量时间都会花费在数据准备上。PyTorch提供了多种辅助工具来帮助用户更方便的处理和加载数据。

- scilit-image:用于读取和处理图片
- pandas:用于解析csv文件



```python
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()
```



### Dataset class

`torch.utils.data.Dataset`实际上是一个用来表示数据集的虚类，自定义数据集需继承Dataset并重写以下方法：

- `__len__`：让自定义数据集支持通过`len(dataset)`来返回dataset的size
- `__getitem__`：让自定义数据集支持通过下标`dataset[i]`来确定第i个数据样本

接下来，尝试创建人脸姿态的自定义数据集，我们将在`__init__`函数读取csv文件，但是会读取图片的逻辑代码写在`__getitem__`方法。

我们数据集样本是字典形式：`{'image':image,'landmarks':landmarks}`。我们的数据集将会接受一个可选参数`transform`，以便可以将任何需要的图片处理操作应用在数据样本上。



### Transform

常用的三种转换操作：

- Rescale：改变图片的尺寸大小
- RandomCrop：对图片进行随机裁剪，数据增强
- ToTensor：将numpy图片转换为tensor数据

写成可调用的类，而不是简单的函数，这样就不必每次调用Transform时都调用参数。为了实现可调用的类, 我们需要实现类的 `__call__` 方法, 并且根据需要实现 `__init__` 方法。

```python
tsfm = Transform(params)
transformed_sample = tsfm(sample)
```



假设我们想把图像的短边重设为256，然后随机裁剪一个224的正方形。

`torchvision`中的`transfomrs.Compose`可以把一系列变换组合起来：

```python
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
 
# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)
 
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)
 
plt.show()
```

`transdorms.Compose`各种变化的组合

`transforms.RandomSizeCrop(224)`随机剪切

`transforms.ToTensor()`把形状为`[H, W, C]`的取值为`[0,255]`的`numpy.ndarray`，转换成形状为`[C, H, W]`，取值为`[0, 1.0]`的`torch.FloadTensor`。

`transforms.Normalize()`最常见的归一化手段，把图像从[0,1]转化成[-1,1]

### Iterating through the dataset

数据采样的过程：

- 从文件中读取一张图片
- 将transforms应用在图片上
- 由于transforms是随机应用的, 因此起到了一定的增广效果

```python
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
 
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
 
    print(i, sample['image'].size(), sample['landmarks'].size())
 
    if i == 3:
        break
```

使用简单的 for 循环遍历数据会丢失很多的可用参数。特别的，如：

- Batching the data
- Shuffling the data
- Load the data in parallel using multiprocessing workers

`torch.utils.data.DataLoader` 是一个提供了所有这些可用参数的迭代器。

`DataLoarder`接受一个`Dataset`类，可以定义batch、size 、shuffle，以及线程个数



### Afterword: torchvision

`torchvision` 包提供了一些共有数据集和转换

`ImageFolder` 是 `torchvision` 中更通用的一个数据集。它假设图片的目录结构如下：

```
    root/ants/xxx.png
    root/ants/xxy.jpeg
    root/ants/xxz.png
    .
    .
    .
    root/bees/123.jpg
    root/bees/nsdf3.png
    root/bees/asd932_.png
```

‘ants’, ‘bees’等是类标签。类似的通用变换也可以在 `PIL.Image` 上运行，如 `RandomHorizontalFlip`, `Scale`

```python
import torch
from torchvision import transforms, datasets
 
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train', 															transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
```

