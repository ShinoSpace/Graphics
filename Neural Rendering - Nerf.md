## Research Groups

## Core Problems in Scholar

## Reviews

### SOTA on Neural Rendering (2020.04)

arxiv: https://arxiv.org/pdf/2004.03805.pdf

### Neural Radiance Fields in 3D Vision (2022.10)

arxiv: https://arxiv.org/pdf/2210.00379.pdf

## Volume Rendering Fundamentals

- [Optical Models for Direct Volume Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf)
- [Scratchapixel - Volume Renderding Equation](https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-summary-equations.html)

### Problems

光栅化算法是将空间中的基本几何形状（三角形）投射到图像上，对云雾、烟烟尘、果冻等非刚体物体渲染效果差

### Overview

A fragment shader：从相机光心出发，向图像上的像素点发射一条射线，光线与射线上碰到的每个材质的粒子群（particles）作用产生颜色，射线上所有计算结果求和即为一个fragment的最终渲染结果

体渲染模型将光线与粒子群的作用分为absorb, emission和scatter，scatter又分为in scatter和out scatter，所有部分求和就是volume rendering equation。Nerf的体渲染方程只考虑absorb和emission，忽略scatter

<center>
<img src="E:/weapons/Graphics/images/research/volume_rendering_buildup.png" width="50%">
</center>

> out-scattering：射线方向上的入射光碰撞粒子群后反射到非射线方向
>
> in-scattering：非射线方向的入射光碰撞粒子群后反射到射线方向

### Absorption (-)

光沿射线方向传播，在$s$处碰到粒子群（介质），在此处取一底面积$A$，高$\Delta s \rightarrow 0$的圆柱体（微元）进行建模。设粒子体密度$\rho(s)$，粒子半径$r$。圆柱体内粒子总数为$\rho(s) A \Delta s$。由于$\Delta s \rightarrow 0$，可以认为所有粒子平铺在圆柱体内部，因此粒子群在圆柱底面上的投影面积为$\rho(s) A \Delta s \pi r^2$。那么光线经过$s$时，单位面积上将有$\rho(s) A \Delta s \pi r^2 / A$的概率碰到粒子，因此可得吸收方程

$$
I(s + \Delta s) = I(s) - I(s)\rho(s) A \Delta s \pi r^2 / A,
\hspace{2pt} \Delta s \rightarrow 0
$$

定义$\sigma(s) = \rho(s) \pi r^2$，这是一个标准的一阶线性齐次微分方程

$$
\begin{gather}
I(s + \Delta s) = I(s) \cdot \left( 1 - \sigma(s) \Delta s \right) \\[5pt]
\frac{dI}{ds} = -I(s) \sigma(s)
\end{gather}
$$

解得

$$
I(b) = I(a)e^{-\int_a^b \sigma(t)dt} \tag{absorption}
$$

#### Density and Transmittance

定义$T(a \rightarrow b) = e^{-\int_a^b \sigma(t)dt}$，$\sigma(s), \hspace{2pt} T(a \rightarrow b)$的物理意义非常重要

- $\sigma(s)$：路径积分$\int_a^b \sigma(t)dt$是光强的某种衰减比例，因此任意单个位置上的$\sigma$就表示粒子群的**密度（density）**，实际上就是粒子面密度。概率解释为：光在$s$处碰到粒子的概率 $\Leftrightarrow \sigma(s)$ is a Probability **Density** Function (PDF)
- $T(a \rightarrow b) = I(b)/I(a)$：相对起始位置$a$，光传播到$b$时光强保留的比例。概率解释是：光从$a$到$b$没有碰到粒子的概率。$T(a \rightarrow b)$称为**透射率**（**transmittance**）或**透明度**（**transparency**）

与透明度正好相反，不透明度**opacity** $= 1 - T(a \rightarrow b)$。不透明度越大，光线保留的越少，对应的颜色分量贡献越少（光线只有**透过**物体才能被观察到）

<center>
<img src="E:/weapons/Graphics/images/research/nerf_opacity_PDF.png" width="50%">
</center>

若粒子群是均匀的（$\text{i.e.} \hspace{2pt} \sigma(s) = constant$），光的辐射强度呈指数衰减，这种特殊情况称为比尔-朗伯吸收定律（Beer-Lambert law）

### Emission (+)

粒子群除了吸收光线，还可能自发光或将其他方向的光线反射到射线方向上

$$
\frac{dI}{ds} = I_e(s)\sigma(s)
$$

### Rendering Equation (Max)

体渲染方程中，absorption和emission两部分的正负号以及积分上下限与坐标系选取有关。在Max的论文[Optical Models for Direct Volume Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf)中，坐标原点在光源处，坐标轴沿射线指向相机光心，因此$\Delta s > 0$时，absorption (-)，emission (+)。Nerf以相机光心为原点，坐标轴指向光源，$\Delta s > 0$时，absorption (+)，emission (-)。本部分建模和求解与Max论文一致，Nerf设定下的求解见[Rendering Equation (Nerf)](#Rendering%20Equation%20(Nerf))

absorption和emission两部分的微元建模之和就是体渲染方程

$$
\frac{dI}{ds} = -I(s)\sigma(s) + I_e(s)\sigma(s)
$$

这是一个一阶线性非齐次微分方程，需要定积分解，因此采用积分因子法求解

> 一阶线性非齐次方程：[Differential Equations](./Fundamental%20Mathematics.md#Differential%20Equations)

积分因子 $= e^{\int_{0}^s \sigma(t)dt}$。设$g(s) = I_e(s) \sigma(s)$，相机与光源的直线距离为$D$。Max的求解目标是$I(D)$，积分下限为0上限为$D$

$$
\begin{gather}
\left( \frac{dI}{ds} + I(s)\sigma(s) \right)e^{\int_{0}^s \sigma(t)dt} =
I_e(s)\sigma(s)e^{\int_{0}^s \sigma(t)dt} \\[2pt]
\int_{s=0}^{s=D} d(I(s)e^{\int_{0}^s \sigma(t)dt}) =
\int_0^D g(s)e^{\int_{0}^s \sigma(t)dt} ds \\[2pt]
I(D)e^{\int_{0}^D \sigma(t)dt} - I(0) = \int_0^D g(s)e^{\int_{0}^s \sigma(t)dt} ds
\end{gather}
$$

渲染方程

$$
I(D) = I(0)e^{-\int_{0}^D \sigma(t)dt} +
\int_0^D I_e(s) \sigma(s)e^{-\int_s^D \sigma(t)dt} ds
$$

## Nerf (n10, i10, e9, s9)

- paper: [Neural Radiance Fields](./papers/NeRF.pdf)
- arxiv: https://arxiv.org/pdf/2003.08934.pdf
- code: https://github.com/bmild/nerf
- supplement material: [Volume Rendering Digest (for NeRF)](./papers/Nerf-supplement.pdf)
- Physical based Rendering Equation: [Optical Models for Direct Volume Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf)
- Riemann Sum: [Volume Rendering Digest (for NeRF)](https://arxiv.org/pdf/2209.02417.pdf)

### Rendering Equation (Nerf)

相机光心为原点，坐标轴指向光源，$\Delta s > 0$时，absorption (+)，emission (-)。设相机光心到光源的直线距离为$D$，改写absorption和emission两个方程

$$
\begin{gather}
\frac{dI}{ds} = I(s)\sigma(s) \tag{absorption} \\[5pt]
\frac{dI}{ds} = -I_e(s)\sigma(s) \tag{emission}
\end{gather}
$$

渲染方程的微分式

$$
\frac{dI}{ds} - I(s)\sigma(s) = -I_e(s)\sigma(s)
$$

求解目标是$I(0)$，积分因子$= e^{-\int_{0}^s \sigma(t)dt}$，积分下限0，上限$D$

$$
\begin{gather}
\left( \frac{dI}{ds} - I(s)\sigma(s) \right)e^{-\int_{0}^s \sigma(t)dt} =
-I_e(s)\sigma(s)e^{-\int_{0}^s \sigma(t)dt} \\[2pt]
\int_{s=0}^{s=D} d(I(s)e^{-\int_{0}^s \sigma(t)dt}) =
-\int_0^D I_e(s)\sigma(s)e^{\int_{0}^s \sigma(t)dt} ds \\[2pt]
I(D)e^{-\int_{0}^D \sigma(t)dt} - I(0) =
-\int_0^D I_e(s)\sigma(s)e^{-\int_{0}^s \sigma(t)dt} ds
\end{gather}
$$

渲染方程

$$
I(0) = I(D)e^{-\int_{0}^D \sigma(t)dt} +
\int_0^D I_e(s)\sigma(s)e^{-\int_{0}^s \sigma(t)dt} ds
$$

$[0, D]$上并非处处都有介质。类似裁剪空间的近平面和远平面，定义射线上的近点$t_n$和远点$t_f$

$$
I(0) = I(D)e^{-\int_{t_n}^{t_f} \sigma(t)dt} +
\int_{t_n}^{t_f} I_e(s)\sigma(s)e^{-\int_{t_n}^{s} \sigma(t)dt} ds
$$

Nerf忽略背景光$I(D)$项。记$T(s) = e^{-\int_{0}^s \sigma(t) dt}$，用颜色$c(s)$替换粒子群自发光辐射强度$I_e(s)$，参数方程$r(t) = o + td$表示射线（$o$为射线起点向量，$d$为射线方向向量），并考虑辐射场规定颜色与观测方向$d$有关

$$
C(r) = \int_{t_n}^{t_f} c(r(t), d)T(t)\sigma(r(t)) dt, \hspace{2pt}
T(t) = e^{-\int_{t_n}^t \sigma(r(s)) ds} \tag{Nerf-RE}
$$

这就是Nerf paper中的渲染方程。由于颜色$c$是替换$I_e$而来，并且忽略了光源项，因此Nerf暗含**物体颜色源于自发光 + 环境光in-scatter的部分**。

### Riemann Sum (piecewise constant data)

#### Fixed sampling problem

$(\text{Nerf-RE})$式没有一般的解析表达，需要在路径上进行采样才能实现，因此需要将积分化为离散求和。离散化的原理非常简单，就是利用定积分的定义，将连续区间按一定间隔划分，然后用不带极限的离散求和近似

$$
\int_{x_0}^{x_n} f(x)dx =
\sum_{i=0}^{n-1} \int_{x_i}^{x_{i + 1}} f(x)dx \approx
\sum_{i=0}^{n-1} f(x_0 + i\Delta x_i)\Delta x_i, \hspace{2pt}
\Delta x_i = x_{i+1} - x_i
$$

积分形式的渲染方程和隐式的3D场景表示的最主要的优势是：**连续**的场景表示。如果每次都等间隔划分积分区间，子区间内固定采样位置（一般取区间端点或中点），用划分得到的这些矩形面积之和来近似渲染方程$(\text{Nerf-RE})$，会在两方面破坏连续性：

- 离散求和会破坏积分的连续性。实现中这是不可避免的，因此只能将间隔取的尽量小
- 神经辐射场的输入$(x, y, z)$固定在采样点处，因此只在这些位置充分训练，非采样点的场景表示能力不足甚至缺失，破坏了连续表示

#### Stratified Sampling

解决方案自然就是非均匀采样：随机划分区间，同时随机选择采样点。Nerf提出stratified sampling（分层采样）策略做非均匀采样

在$[t_n, \hspace{1pt} t_f]$上等间隔划分$N$个子区间，第$i$个子区间为$[t_n + \frac{i - 1}{N}(t_f - t_n), t_n + \frac{i}{N}(t_f - t_n)], \hspace{2pt} i=1, 2, ..., N$，但每个子区间内用均匀分布随机选取采样点

$$
t_i \sim \mathcal{U} \hspace{1pt}
[t_n + \frac{i - 1}{N}(t_f - t_n), t_n + \frac{i}{N}(t_f - t_n)], \hspace{2pt}
i=1, 2, ..., N
$$

当迭代次数足够多时，离散求和就可以无限接近于积分结果，并且神经辐射场所有历史输入形成的空间无限逼近连续空间。$(\text{Nerf-RE})$在第$i$段区间$[t_i, \hspace{1pt} t_{i+1}]$内，$c$和$\sigma$都可近似为常数$c_i, \sigma_i$

$$
C_i = \int_{t_i}^{t_{i+1}} c_i \sigma_i T(t) dt
$$

但$T(t) = e^{-\int_{t_n}^t \sigma(r(s)) ds}$在区间内的变化量不可忽略，因此需要分为两部分

$$
T(t) = e^{-\int_{t_n}^{t_i} \sigma(r(s)) ds} e^{-\int_{t_i}^{t} \sigma_i ds}
$$

> $T(t) = e^{-\int_{t_n}^t \sigma(r(s)) ds}$在子区间内不能视为常数的原因是：$T(t)$是一个从$t_n$开始积分的积分上限函数，在$[t_i, \hspace{1pt} t_{i+1}]$内不满足积分中值定理

这样乘积的后一项积分有解析表达，前一项就是$T(t_i)$

$$
T(t) = T(t_i) e^{-\sigma_i(t - t_i)}
$$

同理，$T(t_i)$也做分段离散化。记$\delta_i = t_{i + 1} - t_i$

$$
T_i = e^{-\sum_{j=1}^{i-1}\int_{t_j}^{t_{j+1}}\sigma(t)dt} =
e^{-\sum_{j=1}^{i-1} \sigma_j\delta_j}
$$

上述结果带回$C_i$

$$
C_i = c_i \sigma_i T_i \int_{t_i}^{t_{i + 1}} e^{-\sigma_i(t - t_i)} dt =
c_i T_i \left(1 - e^{-\sigma_i \delta_i} \right)
$$

所有$C_i$求和就是离散化的渲染方程

$$
C = \sum_{i=1}^{N} T_i c_i (1 - e^{-\sigma_i \delta_i}), \hspace{3pt}
\text{where} \hspace{3pt} T_i = e^{-\sum_{j=1}^{i-1} \sigma_j\delta_j}
$$

#### Consistency

观察同一视角下相邻的两次优化：两组不同的随机采样点，渲染方程计算出两张不同的渲染图，但他们都以同一张图为优化目标，因此**模型对采样具有一致性**。

### Positional Encoding

#### Motivation: Low frequency bias/prior of Deep Network

最原始的Nerf渲染出的图比较糊，在颜色、几何边界处效果差，说明模型拟合高频信息的能力较差。

[Spectral Bias](https://arxiv.org/pdf/1806.08734.pdf), [Frequency Bias](https://arxiv.org/pdf/2003.04560.pdf), [Fourier Features](https://arxiv.org/pdf/2006.10739.pdf)等理论工作证明：如果不施加外力，Deep Networks**倾向于**学出一个低频函数（low frequency function），但只是**倾向**，这可以看作是Deep Network的一个先验。如果将输入用一个高频函数映射到高维空间（位置编码就是一种高频函数），MLP同样可以学习高频信息

<center>
<img src="E:/weapons/Graphics/images/research/nerf-PE-1.png" width="50%"><br>
<img src="E:/weapons/Graphics/images/research/nerf-PE-2.png" width="50%">
</center>

直观上，**DeepNet输入数据的性质直接影响模型学习的倾向**。如果输入数据的性质/分布与预期的输出不匹配，模型学习效果就会变差。直接给MLP输入$(x, y, z, \theta, \phi)$，相邻位置变化不明显（慢变、低频），MLP就较难学好高频变化。

### Hierarchical Scene Representation

射线上采样，采了很多空气，但很显然有物体的位置是更重要的，因此需要重要性采样。重要性采样也是Ray Tracing中的重要部分。

在stratified sampling基础上，NeRF叠加了一种由粗到细的（coarse-to-fine）场景表示：训练coarse和fine两个场景表示网络。coarse network直接在射线上stratified sample $N_c$个点，计算渲染方程（$hat$表示模型预测值）

$$
\hat{C}_c(r) = \sum_{i=1}^{N_c} w_i c_i,
\hspace{2pt} w_i = T_i(1 - e^{-\sigma_i \delta_i})
$$

这里将渲染方程看作光路上所有采样位置颜色的加权和，权重$w_i$为$\alpha_i = 1 - e^{-\sigma_i \delta_i}$与透射率$T_i$的乘积，但权重不满足归一化条件$\sum_i w_i = 1$，因此再做一步归一化就得到在光路上划分的分段区间上的离散分布$\hat{w}_i = w_i / \sum_{j=1}^{N_c}$（$\hat{w}_i$就类似于alpha compositing的加权系数）。用逆变换采样（Inverse Transform Sampling）根据分布$\hat{w}_i$再采样$N_f$个位置。无论这些位置上是否有物体，都是重点学习的区域（有物体，肯定要多学。没物体，说明是个错误的预测，更要着重学，相当于hard negative mining）。$N_c + N_f$送进fine network进行学习。

> 逆变换采样：[Inverse Transform Sampling](./Fundamental%20Mathematics.md#Inverse%20Transform%20Sampling)

直观上，Inverse Transform Sampling采样的$N_f$个位置就是光路上存在物体的区域，是需要重点训练的位置，因此设置$N_f > N_C$。

### NeRF的问题

#### 光路独立，同一个物体不同光路缺乏关联性

同一个物体，距离较近的光路与物体的碰撞点之间缺乏关联。直觉上，物体上相近的两个点在camera下的depth有可能非常接近。

#### 神经辐射场的信息复合度较高，训练困难

神经辐射场中同时隐式地存储了材质密度（density）、颜色，并且为了表示3D场景，很可能需要一些辅助信息。

### NeRF's working flow & Details

还差三件事：

1. llff数据处理看完
2. render过程，ndc已经搞定了，之后的看完
3. data preprocess和完整的render过程（包含ray生成、坐标变换以及模型训练方式等）分为两部分，画两个流程图，每个流程图内要模块化，重要模块给note（例如NDC）

## Behind the Scenes (n7, e7, s8)

- arxiv: https://arxiv.org/pdf/2301.07668.pdf
- code: https://github.com/Brummi/BehindTheScenes
- group: 慕尼黑工业大学（TUM）

### Problems

只用单张图做geometry level的3D场景重建。传统的per-pixel depth estimation方法，每个像素点只能出一个depth，无法处理透视视角下的部分遮挡问题

### Brainstorm

这篇文章的原始动机还是为了改进NeRF：神经辐射场同时要学习物体的density和颜色，负担比较重

- density建模物体材质的透光率，影响物体表面的颜色。对于三维重建来说，density可以直接描述物体的几何信息，以及空间中的位置是否被占据（occupancy），是3D重建最关键的部分之一
- 颜色：最直接的想法，可以直接从图像中取颜色。虽然着色受视角影响，但离得近的不同视角下颜色是相近的。直接取颜色能极大降低模型学习难度

因此核心想法是：神经辐射场只负责从图像中建模density，颜色直接从其他视角取。相似的视角下颜色相差不大，因此需要使用视频数据

### Highlights

1. occupancy仍然使用神经辐射场，没有盲目将隐式表示换为网格表示

    隐式表示的最大优势就是连续性。神经辐射场隐式表示连续的3D场景空间，常规的occupancy grid可以直接采样神经辐射场生成。直接在occupancy grid上进行体渲染不是最优解

2. 颜色不必预测，可以复用视角相似的图像

    这一点是比较显然的，只要视角相差不大，颜色差别很小，通过pose直接投影取值即可

3. Per-pixel reprojection loss能够自适应地处理遮挡问题

    训练时选取$k$张「参考图」取颜色，在新视角下分别渲染出$k$张图。计算损失时，只让取到正确颜色的pixel对训练有贡献，完成这件事的方法是*per-pixel minmum*：$k$张渲染出的图与监督图像分别计算逐像素的loss（不做reduction），在$(u, v)$位置的像素有$k$个loss，最小的loss对参数优化产生贡献。这种方法自适应地处理了遮挡问题

    <center>
    <img src="E:/weapons/Graphics/images/research/per-pixel_reproj_loss.png" width="65%">
    </center>

    这个损失函数源于[Monodepth2 (ICCV19)](./papers/monodepth2-ssl.pdf)，这个工作用自监督学习做训练，在当时是非常超前的

    <center>
    <img src="E:/weapons/Graphics/images/research/monodepth2.png" width="65%">
    </center>

4. density必须抑制错误的颜色

    <center>
    <img src="E:/weapons/Graphics/images/research/behind-the-scene.png" width="30%">
    </center>
