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

> out-scattering：射线方向上的入射光碰撞粒子群后反射到非射线方向
> in-scattering：非射线方向的入射光碰撞粒子群后反射到射线方向

#### Absorption (-)



#### Emission (+)



### Nerf

- Paper: https://arxiv.org/pdf/2003.08934.pdf
- Code: https://github.com/bmild/nerf

