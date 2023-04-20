## Linear Algebra

## Geometry of Linear Transformation

> 改编自3Blue1Brown系列视频：线性代数的本质

- 线性变换：$\vec{y} = A \vec{x}$
- 应用：点或向量的坐标变换，换系
- 核心：基变换

### 线性变换

#### 线性 or 刚体坐标变换的本质是基变换

- 基向量完备地描述了向量空间
- 变换基向量 $\Leftrightarrow$ 对空间内所有点/向量进行变换

以笛卡尔坐标系下的旋转变换为例

原始基向量为$u=(1,0)$, $v=(0,1)$。给定任意向量$x$，**将向量和空间视作一个整体**。将空间（基向量）顺时针旋转$\theta$角（记旋转后的空间的基向量为$u^\prime, v^\prime$），$x$也就旋转了同样的角度（旋转后的向量记为$y$）。旋转过程中，向量与空间并未发生相对运动，因此，$x$和$y$在旋转前后的两个空间内，**坐标保持不变**。

求旋转后的向量在原空间$u, v$下的坐标，只需要知道**旋转后的空间的基向量$u^\prime, v^\prime$在原空间下的坐标**即可。由于$x$和$y$在原空间和旋转空间下的坐标相同，因此直接线性组合即得所求

$$ y = \begin{pmatrix}
u^\prime, v^\prime
\end{pmatrix} x = Ax $$

### 坐标系转换

#### 换系就是直接换基

任意给定向量$a$在原空间$u^\prime, v^\prime$下的坐标为$a = (x, y)$，求$a$在目标空间$u, v$下的坐标，只需**求出$u^\prime, v^\prime$在目标空间$u, v$下的坐标**，然后线性组合即可。

### 线性变换矩阵的几何意义

- 原空间：向量 or 点在这个空间下的坐标已知或容易求解
- 目标空间（参考空间）：求向量 or 点在这个空间下的坐标

$A$: 原空间的基向量$u^\prime, v^\prime$在目标空间$u, v$下的坐标，即$A=(u^\prime, v^\prime)$

--------------------

## Calculus

## Differential Equations

### 一阶线性齐次

$$ \begin{gather}
y' + p(x)y = 0 \Rightarrow
\frac{1}{y}dy = -p(x)dx \\[3pt]
y = Ce^{-\int p(x)dx} \tag{1}
\end{gather} $$

### 一阶线性非齐次

$$ y' + p(x)y = q(x) \tag{2} $$

#### 欧拉 - 拉格朗日常数变易法

常数变易法是一类微分方程求解的通法。该方法的结果是不定积分形式，不擅长处理隐式表达的$p(x), q(x)$定积分解。

令$(1)$中$C \rightarrow c(x)$，$y = c(x)e^{-\int p(x)dx}$，然后带入方程求解$c(x)$

$$ \begin{gather}
y' = c'(x)e^{-\int p(x)dx} - c(x)p(x)e^{-\int p(x)dx} \\[5pt]
c'(x)e^{-\int p(x)dx} = q(x) \Rightarrow c(x) = \int e^{\int p(x)dx} q(x) dx + C \\[5pt]
y = e^{-\int p(x)dx} \left( \int e^{\int p(x)dx} q(x) dx + C \right)
\end{gather} $$

#### 积分因子

该方法能够直接对方程进行积分求解，对$p(x), q(x)$的表达形式没有限制，因此能够直接结合初值进行定积分。

最主要的motivation是，$(2)$式左侧的形式与$(uv)'$形式相似，仅$y'$项缺少了与某个函数的乘积。因此希望构造一个$z(x)$乘到$(2)$式上，使得方程左侧能够凑成$(zy)'$

$$ zy' + zpy = zq $$

我们目标是让上式左侧成为$(zy)'=zy' + z'y$的形式，那么自然就用待定系数法

$$ zy' + z'y = zy' + zpy \Rightarrow z' - zp = 0 $$

这显然是一个一阶线性齐次方程，立刻推$z(x) = Ce^{\int_*^x p(t)dt}$，取一个特解$z(x) = e^{\int_{x_0}^x p(t)dt}$即可。这里将$z(x)$写为变上限积分形式，便于后续求解。另外，$z(x)$的积分下限可以根据具体问题的初值条件任意选定，这样可以使最终结果更简单。方程$(2)$乘以这个积分因子$z(x)$，左侧就可以凑成$(uv)'$的形式

$$ \begin{gather}
e^{\int_{x_0}^x p(t)dt} y' + e^{\int_{x_0}^x p(t)dt} p(x) y = q(x)e^{\int_{x_0}^x p(t)dt} \\[5pt]
\frac{d(e^{\int_{x_0}^x p(t)dt}y)}{dx} = e^{\int_{x_0}^x p(t)dt} q(x) \\
\end{gather} $$

上式可以很清晰地做（带初值的）定积分

$$ \begin{gather} \int_{x=x_0}^{x=x_1} d(e^{\int_{x_0}^x p(t)dt}y) =
\int_{x_0}^{x_1} e^{\int_{x_0}^x p(t)dt} q(x) dx
\end{gather} $$

积分因子法对于这类方程的求解过程很清晰，定积分和不定积分都容易处理。Volume Rendering渲染方程的推导就是用的这种方法。

--------------------

## Probability and Statistics

## Random Variable Sampling

设随机变量$X$满足某种分布，PDF记为$f_X(x)$，CDF记为$F_X(x)$。当采样次数足够多时，所有采样点的分布应该满足CDF。

随机采样是应用性非常强的技术，合理的学习路线是：先理解怎么做，然后给足够的例子做应用练习，以此形成数学物理直觉，最后考虑数学证明。

参考资料：[Wikipedia](https://en.wikipedia.org/)

### Inverse Transform Sampling

参考：[Inverse transform sampling - Wikipedia](https://en.wikipedia.org/wiki/Inverse_transform_sampling)

#### Method

设CDF可逆，逆函数为$x = F^{-1}(y) = \inf\{x|F(x) \geq y\}$，目标是采样$N$个点

1. 用均匀分布$U(0, 1)$采样$N$个值$u_i$
2. 计算逆函数值$x_i = F^{-1}(u_i)$

$x_i$就是目标采样点，与$X$同分布。

对于离散分布$F_X(x) = P(X \leq x) = \sum_{i} p_i$，CDF的值域是离散的累积概率。当均匀分布的值落在非离散值上时，对应的$x$是没有定义的。观察CDF图像，直觉告诉我们只需要取累积概率跳变处对应的$x$即可

<center>
<img src="E:/weapons/Graphics/src/research/inverse_transform_sampling_discrete_dist.png" width="50%">
</center>

这样做显然是对的：若依分布$F_X(x)$采样，当采样点数足够多时，$x_i$出现的比例应当为$p_i$，而上述取法恰好满足这个要求。如果$u_i$恰好落在累计概率上，$x$就有两个可能的取值。由于有限数量的点不影响整体的分布，因此取哪一个都可以（但要保证规则的一致性）

#### Examples

指数分布

<center>
<img src="E:/weapons/Graphics/src/research/inverse_transform_sampling_exp.jpg" width="30%">
</center>

