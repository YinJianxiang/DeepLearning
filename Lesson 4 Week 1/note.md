卷积层(conv)和池化层(pool)

$n_h$、$n_w$和$n_c$代表图像的高度、宽度和通道数。



![model.png](https://img-blog.csdn.net/2018042521470222?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM3MzMzMjY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

卷积模块

- 使用0扩充边界
- 卷积窗口
- 前向卷积
- 反向卷积

池化模块

- 前向池化
- 创建掩码
- 值分配
- 反向池化

卷积层反向传播

dA
$$
dA += \sum_{h=0}^{n_{H}}\sum_{w=0}^{n_{W}} W_{c} \times dZ_{hw} 
$$
$W_{c}$过滤器，$Z_{hw}$是卷积层第h行第w列的使用点乘计算后的输出Z的梯度。
$$
dW_{c} += \sum_{h=0}^{n_{H}} \sum{w=0}^{n_{W}} a_{slice} \times dZ_{hw}
$$


池化层反向传播

average
$$
dZ = 1  
dZ = \left[ \begin{matrix}
\frac{1}{4} & \frac{1}{4} \\
\frac{1}{4} & \frac{1}{4}
\end{matrix}
\right]
$$
输入梯度 += dz 

max

创建掩码
$$
X = \left[ \begin{matrix}
3 & 4 \\
5 & 2
\end{matrix}\right]

M = \left[ \begin{matrix}
0 & 0 \\
1 & 0
\end{matrix}\right]
$$
输入梯度 += 输出梯度*掩码



使用的CNN框架

- Conv2d：stride:1 padding:"same"
- ReLU
- Max pool:f:8 stride:8 padding:"same"
- Conv2d: stride:1 padding:"same"
- Relu
- Max pool:f:4 stride:4 padding:"same"
- 全连接层(FC):

