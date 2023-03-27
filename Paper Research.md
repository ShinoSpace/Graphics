## Neural Rendering

- Summary/Review: https://arxiv.org/abs/2004.03805

### Volume Rendering Fundamentals

#### Papers & Materials

- [Optical Models for Direct Volume Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf)
- [Volume Rendering Digest (for NeRF)](https://arxiv.org/pdf/2209.02417.pdf)
- [Scratchapixel - Volume Renderding Equation](https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-summary-equations.html)

#### Problem & Motivation

光栅化算法是将空间中的基本几何形状（三角形）投射到图像上，对云雾、烟烟尘、果冻等非刚体物体渲染效果差

#### Overview

A fragment shader：从相机光心出发，向图像上的像素点发射一条射线，光线与射线上碰到的每个材质的粒子群（particles）作用产生颜色，射线上所有计算结果求和即为一个fragment的最终渲染结果

体渲染模型将光线与粒子群的作用分为absorb, emission和scatter，scatter又分为in scatter和out scatter，所有部分求和就是volume rendering equation。Nerf的体渲染方程只考虑absorb和emission，忽略scatter

<div align=center>
<img src="E:/weapons/Graphics/src/paper-research/volume_rendering_buildup.png" width="50%">
</div>

> out-scattering：射线方向上的入射光碰撞粒子群后反射到非射线方向
> in-scattering：非射线方向的入射光碰撞粒子群后反射到射线方向

#### Absorption (-)

光沿射线方向传播，在$s$处碰到粒子群（介质），在此处取一底面积$A$，高$\Delta s \rightarrow 0$的圆柱体（微元）进行建模。设粒子体密度$\rho(s)$，粒子半径$r$。圆柱体内粒子总数为$\rho(s) A \Delta s$。由于$\Delta s \rightarrow 0$，可以认为所有粒子平铺在圆柱体内部，因此粒子群在圆柱底面上的投影面积为$\rho(s) A \Delta s \pi r^2$。那么光线经过$s$时，单位面积上将有$\rho(s) A \Delta s \pi r^2 / A$的概率碰到粒子，因此可得吸收方程

$$ I(s + \Delta s) = I(s) - I(s)\rho(s) A \Delta s \pi r^2 / A,
\hspace{2pt} \Delta s \rightarrow 0
$$

定义$\sigma(s) = \rho(s) \pi r^2$，这是一个标准的一阶线性齐次微分方程

$$ \begin{gather}
I(s + \Delta s) = I(s) \cdot \left( 1 - \sigma(s) \Delta s \right) \\[5pt]
\frac{dI}{ds} = -I(s) \sigma(s) \tag{*}
\end{gather} $$

解得

$$ I(b) = I(a)e^{-\int_a^b \sigma(t)dt} $$

定义$T(a \rightarrow b) = e^{-\int_a^b \sigma(t)dt}$，$\sigma(s), T(a \rightarrow b)$的物理意义非常重要

$\sigma(s)$：光在$s$处传播极小一段距离$ds$碰到粒子的概率

> TODO: 随便画一个$I(s)$的图，考虑(\*)式的在图像上的含义

若粒子群是均匀的（$\sigma(s)$为常数），入射光进入粒子群后，辐射强度呈指数衰减，这种特殊情况称为比尔-朗伯吸收定律（Beer-Lambert law）<br>
$T(a \rightarrow b)$：$T(a \rightarrow b) = I(b)/I(a)$，光沿射线从$a$传播到$b$，没有碰到粒子的概率，称为光的**透射率（Transmittance）**

#### Emission (+)

粒子群除了吸收光线，还可能自发光或将其他方向的光线反射到射线方向上

$$ \frac{dI}{ds} = I_e(s)\sigma(s) $$

#### Rendering Equation (Max)

体渲染方程中，absorption和emission两部分的正负号以及积分上下限与坐标系选取有关。在Max的论文[Optical Models for Direct Volume Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf)中，坐标原点在光源处，坐标轴沿射线指向相机光心，因此$\Delta s > 0$时，absorption (-)，emission (+)。Nerf以相机光心为原点，坐标轴指向光源，$\Delta s > 0$时，absorption (+)，emission (-)。本部分建模和求解与Max论文一致，Nerf设定下的求解见[Rendering Equation (Nerf)](#Rendering%20Equation%20(Nerf))

absorption和emission两部分的微元建模之和就是体渲染方程

$$ \frac{dI}{ds} = -I(s)\sigma(s) + I_e(s)\sigma(s) $$

这是一个一阶线性非齐次微分方程，需要定积分解，因此采用积分因子法求解

> see Calculus in [Fundamental Calculus and Linear Algebra](./Fundamental%20Calculus%20and%20Linear%20Algebra.md)

积分因子 $= e^{\int_{0}^s \sigma(t)dt}$。设$g(s) = I_e(s) \sigma(s)$，相机与光源的直线距离为$D$。Max的求解目标是$I(D)$，积分下限为0上限为$D$

$$ \begin{gather}
\left( \frac{dI}{ds} + I(s)\sigma(s) \right)e^{\int_{0}^s \sigma(t)dt} =
I_e(s)\sigma(s)e^{\int_{0}^s \sigma(t)dt} \\[2pt]
\int_{s=0}^{s=D} d(I(s)e^{\int_{0}^s \sigma(t)dt}) =
\int_0^D g(s)e^{\int_{0}^s \sigma(t)dt} ds \\[2pt]
I(D)e^{\int_{0}^D \sigma(t)dt} - I(0) = \int_0^D g(s)e^{\int_{0}^s \sigma(t)dt} ds
\end{gather} $$

渲染方程

$$ I(D) = I(0)e^{-\int_{0}^D \sigma(t)dt} +
\int_0^D g(s)e^{-\int_s^D \sigma(t)dt} ds $$

### Nerf (n10, i10, e9, s9)

- Paper: https://arxiv.org/pdf/2003.08934.pdf
- Code: https://github.com/bmild/nerf
- Physical based Rendering Equation: [Optical Models for Direct Volume Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf)
- Riemann Sum: [Volume Rendering Digest (for NeRF)](https://arxiv.org/pdf/2209.02417.pdf)

#### Rendering Equation (Nerf)

相机光心为原点，坐标轴指向光源，$\Delta s > 0$时，absorption (+)，emission (-)。设相机光心到光源的直线距离为$D$，改写absorption和emission两个方程

$$ \begin{gather}
\frac{dI}{ds} = I(s)\sigma(s) \tag{absorption} \\[5pt]
\frac{dI}{ds} = -I_e(s)\sigma(s) \tag{emission}
\end{gather} $$

渲染方程的微分式

$$ \frac{dI}{ds} - I(s)\sigma(s) = -I_e(s)\sigma(s) $$

求解目标是$I(0)$，积分因子$= e^{-\int_{0}^s \sigma(t)dt}$，积分下限0，上限$D$

$$ \begin{gather}
\left( \frac{dI}{ds} - I(s)\sigma(s) \right)e^{-\int_{0}^s \sigma(t)dt} =
-I_e(s)\sigma(s)e^{-\int_{0}^s \sigma(t)dt} \\[2pt]
\int_{s=0}^{s=D} d(I(s)e^{-\int_{0}^s \sigma(t)dt}) =
-\int_0^D I_e(s)\sigma(s)e^{\int_{0}^s \sigma(t)dt} ds \\[2pt]
I(D)e^{-\int_{0}^D \sigma(t)dt} - I(0) =
-\int_0^D I_e(s)\sigma(s)e^{-\int_{0}^s \sigma(t)dt} ds
\end{gather} $$

渲染方程

$$ I(0) = I(D)e^{-\int_{0}^D \sigma(t)dt} +
\int_0^D I_e(s)\sigma(s)e^{-\int_{0}^s \sigma(t)dt} ds $$

$[0, D]$上并非处处都有介质。类似裁剪空间的近平面和远平面，定义射线上的近点$t_n$和远点$t_f$

$$ I(0) = I(D)e^{-\int_{t_n}^{t_f} \sigma(t)dt} +
\int_{t_n}^{t_f} I_e(s)\sigma(s)e^{-\int_{t_n}^{s} \sigma(t)dt} ds $$

Nerf忽略背景光$I(D)$项。记$T(s) = e^{-\int_{0}^s \sigma(t) dt}$，用颜色$c(s)$替换辐射强度，用参数方程$r(t) = o + td$表示射线（$o$为射线起点向量，$d$为射线方向向量），并考虑辐射场规定颜色与观测方向$d$有关

$$ C(r) = \int_{t_n}^{t_f} c(r(t), d)T(t)\sigma(r(t)) dt, \hspace{2pt}
T(t) = e^{-\int_{t_n}^t \sigma(r(s)) ds} \tag{Nerf-RE} $$

这就是Nerf的渲染方程

#### Riemann Sum (Nerf, piecewise constant data)

$(\text{Nerf-RE})$式没有一般的解析表达，需要在路径上进行采样才能实现，因此需要将积分化为离散求和。离散化的原理非常简单，就是利用定积分的定义，将连续区间按一定间隔划分，然后用不带极限的离散求和近似

$$ \int_{x_0}^{x_1} f(x)dx \approx \sum_{i} f(x_0 + i\Delta x_i)\Delta x_i $$

如果每次都等间隔划分积分区间，子区间内固定采样位置（一般取区间端点或中点），用划分得到的这些矩形面积之和来近似$(\text{Nerf-RE})$，会破坏场景表示的连续性。因此区间划分需要具有随机性。Nerf的做法是，仍然在$[t_n, \hspace{1pt} t_f]$上等间隔划分$N$个子区间，第$i$个子区间为$[t_n + \frac{i - 1}{N}(t_f - t_n), t_n + \frac{i}{N}(t_f - t_n)], \hspace{2pt} i=1, 2, ..., N$，但每个子区间内用均匀分布随机选取采样点

$$ t_i \sim \mathcal{U} \hspace{1pt}
[t_n + \frac{i - 1}{N}(t_f - t_n), t_n + \frac{i}{N}(t_f - t_n)], \hspace{2pt}
i=1, 2, ..., N
$$

当迭代次数足够多时，离散求和就可以无限接近于积分结果。$(\text{Nerf-RE})$在第$i$段区间$[t_i, \hspace{1pt} t_{i+1}]$内，$c$和$\sigma$都可近似为常数$c_i, \sigma_i$

$$ C_i = \int_{t_i}^{t_{i+1}} c_i \sigma_i T(t) dt $$

但$T(t) = e^{-\int_{t_n}^t \sigma(r(s)) ds}$在区间内的变化量不可忽略，因此需要分为两部分

$$ T(t) = e^{-\int_{t_n}^{t_i} \sigma(r(s)) ds} e^{-\int_{t_i}^{t} \sigma_i ds} $$

> $T(t) = e^{-\int_{t_n}^t \sigma(r(s)) ds}$在子区间内不能视为常数的原因是：$T(t)$是一个从$t_n$开始积分的积分上限函数，在$[t_i, \hspace{1pt} t_{i+1}]$内不满足积分中值定理

这样乘积的后一项积分有解析表达，前一项就是$T(t_i)$

$$ T(t) = T(t_i) e^{-\sigma_i(t - t_i)} $$

同理，$T(t_i)$也做分段离散化。记$\delta_i = t_{i + 1} - t_i$

$$ T_i = e^{-\sum_{j=1}^{i-1} \sigma_j\delta_j} $$

上述结果带回$C_i$

$$ C_i = c_i \sigma_i T_i \int_{t_i}^{t_{i + 1}} e^{-\sigma_i(t - t_i)} dt =
c_i T_i \left(1 - e^{-\sigma_i \delta_i} \right) $$

所有$C_i$求和就是离散化的渲染方程

$$ C = \sum_{i=1}^{N} T_i c_i (1 - e^{-\sigma_i \delta_i}), \hspace{3pt}
\text{where} \hspace{3pt} T_i = e^{-\sum_{j=1}^{i-1} \sigma_j\delta_j} $$

