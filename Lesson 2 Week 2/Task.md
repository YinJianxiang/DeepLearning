主要任务：

1. 分割数据集

2. 优化梯度下降

   2.1 不适用任何优化

   2.2 mini-batch梯度下降法

   2.3 使用具有动量的梯度下降算法

   2.4 使用Adam算法

**Adam算法**
$$
v_{dW^{[l]}} = \beta_{l} v_{dW^{[l]}} + (1 - \beta_{1})\frac{\partial J}{\partial W^{[l]}}
$$

$$
v^{corrected}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - (\beta_{1})^{t}}
$$

$$
s_{dW^{[l]}} = \beta_{2} s_{dW^{[l]}} + (1 - \beta_{2})(\frac{\partial J}{\partial W^{[l]}})^{2}
$$

$$
s^{corrected}_{dW^{[l]}} = \frac{s_{dW^{[l]}}}{1 - (\beta_{2})^{t}}
$$

$$
W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}{\sqrt{s^{corrected}_{dW^{[l]}} + \varepsilon}}
$$

其中的参数:

$t$：当前迭代次数

$l$：当前神经网络的层数
$\beta_1$和$\beta_2$ :控制两个指数加权平均值的超参数
$\alpha$ :学习率
$\varepsilon$ ：一个非常小的数，用于避免除零操作，一般为 $1^{-8}$
