### 参考资料

- Mathematics: 线性代数，3Blue1Brown (3B1B)
- Course: GAMES101
- Textbook: Fundamental of Computer Graphics $\geq$ 3rd edition

--------------------

### 数学基础：理解线性变换的几何意义

> 改编自3Blue1Brown系列视频：线性代数的本质

- 线性变换：$\vec{y} = A \vec{x}$
- 应用：点或向量的坐标变换，换系
- 核心：基变换

#### 坐标变换的本质是基变换

- 基向量完备地描述了向量空间
- 变换基向量 $\Leftrightarrow$ 对空间内所有点/向量进行变换

以笛卡尔坐标系下的旋转变换为例

原始基向量为$u=(1,0)$, $v=(0,1)$。给定任意向量$x$，将向量和空间视作一个整体。将空间（基向量）顺时针旋转$\theta$角（记旋转后的空间的基向量为$u^\prime, v^\prime$），$x$也就旋转了同样的角度（旋转后的向量记为$y$）。旋转过程中，向量与空间并未发生相对运动，因此，$x$和$y$在旋转前后的两个空间内，**坐标保持不变**。

求旋转后的向量在原空间$u, v$下的坐标，只需要知道旋转后的空间的基向量$u^\prime, v^\prime$在原空间下的坐标即可。由于$x$和$y$在原空间和旋转空间下的坐标相同，因此直接线性组合即得所求

$$ y = \begin{pmatrix}
u^\prime, v^\prime
\end{pmatrix} x = Ax $$

#### 换系是线性变换本质的直接应用

- 换系就是在换基

给定任意向量$x$，已知其在原空间$u^\prime, v^\prime$下的坐标，求$x$在目标空间$u, v$下的坐标。只要求得原空间的基$u^\prime, v^\prime$在目标空间$u, v$下的坐标，直接线性组合即得所求。

#### 线性变换矩阵A的几何意义

明确两个空间

- 目标空间：求向量 or 点在这个空间下的坐标
- 已知空间：向量 or 点在这个空间下的坐标已知或容易求解

$A$: 已知空间的基向量$u^\prime, v^\prime$在目标空间$u, v$下的坐标，即$A=(u^\prime, v^\prime)$

--------------------

### Transformation and Homogeneous coordinate

- prerequisite: [线性变换的几何意义](#数学基础理解线性变换的几何意义)，根据线性变换的本质快速确定变换矩阵

#### 齐次坐标（Homogeneous coordinate）

- 为什么引入齐次坐标？

  适配平移。引入齐次坐标后，线性变换和平移可以直接组合成一个矩阵表示。「连续应用多个变换 $\Leftrightarrow$ 变换矩阵连乘」这一性质保持不变

- 齐次坐标对向量和点的定义不同

  vector：$v = (x, y, 0)^T$<br>
  point：$P(x, y, 1)$

  这种定义的目的是保证代数运算后的结果与定义的一致性。注意：最后一维不能理解为常规3维空间的Z方向，因为其进行的是非常规的代数运算。

- 向量和点的运算，以及几何含义

  为什么这样定义向量和点？-> 保证运算结果与定义的一致性！

  - vector + vector = vector：常规向量加法，得到和向量，最后一维恒为0<br>
  - point - point = vector: 终点B - 起点A，得到向量$\overrightarrow{AB}$<br>
  - point + vector = point: 求和后最后一维为1，说明结果是一个点的坐标。几何意义：从给定点出发，沿向量方向行走向量长度距离，到达另一点。<br>
  - point + point: 直观上没有意义。在齐次坐标下扩充定义。

- 点的扩充定义

  给定点的齐次坐标，最后一维为$w$，则其表示的点为$(x, y, w) = (x/w, y/w, 1)$

  在扩充定义下，两个点的和为$(x_1, y_1, 1) + (x_2, y_2, 1) = (\frac{x_1+x_2}{2}, \frac{y_1+y_2}{2}, 1)$，几何上表示这两个点的中点坐标。

#### 旋转方向的选取对旋转矩阵的影响

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rigid%20transform/rotation_angle.png" width="50%"  height="50%">
</div>

- Why is the rotation matrix $R_y(\alpha)$ around the y-axis different?

  取决于旋转角方向的选取。

  二维直角坐标下，顺时针和逆时针旋转的旋转矩阵出现与之相同的现象

  这里的$R_y(\alpha)$，旋转角是从$z$到$x$的。**该方向符合右手螺旋法则，一般将其定义为旋转的正方向**。

#### 绕过原点的任意轴旋转：罗德里格斯公式

证明思路有两种：

1. 将$x/y/z$三个轴旋转到给定的任意轴（换系），然后执行绕$x/y/z$旋转$R_x(\alpha)/R_y(\alpha)/R_z(\alpha)$，最后再转回来。这种方法比较麻烦，并且可以预见到，最后结果是若干个矩阵的连乘，化简会很困难
2. 向量绕轴旋转的本质：只有垂直于旋转轴的方向发生旋转。因此将向量在两个方向上分解：垂直于旋转轴的方向（旋转实际发生的方向）和平行于旋转轴的方向（该方向上不发生任何变化）

proof:

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rigid%20transform/Rodrigues_rotation_proof_0.png" width="28%"> <img src="E:/weapons/Graphics/src/games101/rigid%20transform/Rodrigues_rotation_proof_1.png" width="32.6%"> <img src="E:/weapons/Graphics/src/games101/rigid%20transform/Rodrigues_rotation_proof_2.png" width="36%">
</div>

Assume:

- 任意向量$\vec{s}$
- 任意旋转轴方向向量$\hat{a}$
- 绕轴旋转角$\theta$

**Step1** 将$\vec{s}$分解为垂直于旋转轴（旋转方向）和平行于旋转轴两个分量$s_{\kern{-1pt}/\kern{-2pt}/}, s_{\perp}$

**Step2**  $\vec{s}$绕轴$\hat{a}$旋转$\theta$角，只需$s_{\perp}$旋转$\theta$角即可。为了表示旋转后的垂直分量$s_{\perp}^{ROT}$，需要一个$\hat{a}, \hat{b}$构成的平面垂直的轴（相当于建立了另一个直角坐标系来完备地描述任意向量）。这个轴很容易得到，$\hat{a}$和$\hat{b}$直接叉乘即可$\hat{c} = \hat{a} \times \hat{b}$

**Step3** 将$s_{\perp}$旋转$\theta$角得到$s_{\perp}^{ROT}$，并与$s_{\kern{-1pt}/\kern{-2pt}/}$求和即可

--------------------

### Viewing Transformation

- View/Camera/ModelView Transformation: 物体和相机同时运动（不发生相对运动）
  - 与换系区分：物体不动，换到另一个坐标系，观测者（相机）与物体发生了相对运动
- Projection Transformation
  - Orthographic projection（正交投影）
  - Perspective projection（透视投影）

ModelView + Projection两种变换合称为MVP变换

#### ModelView Transformation

确定相机的位置和姿态，需要：相机中心$\vec{e} = (x, y, z)$，相机的look at/gaze方向$\hat{g}$（朝哪儿看），向上方向$\hat{t}$（相机是正着拍，还是斜着拍？）

相对运动：**如果相机和物体、场景不发生相对运动，那么无论怎么移动、旋转，拍出来的东西都是一模一样的**。那么干脆将相机（坐标系）移动到一个标准位置，方便分析问题。

<div align=center>
<img src="E:/weapons/Graphics/src/games101/MVP%20Transform/camera_pose.png" width="50%">
</div>

**ModelView Transformation**：**我们站在标准坐标系下**（上图$X$-$Y$-$Z$坐标系）

1. 将相机中心$\vec{e}$移动至$(0, 0, 0)$
2. 相机look at方向旋转至$-z$：$g \rightarrow (0, 0, -1)$
3. 向上方向$\hat{t}$旋转到$Y$轴：$\hat{t} \rightarrow (0, 1, 0)$。旋转后$t \times (-g)$（即$Y \times Z$）自然与$X$轴重合

所有物体也同时进行上述变换，将上述变换矩阵记为$M_{view}$

<div align=center>
<img src="E:/weapons/Graphics/src/games101/MVP%20Transform/camera_pose_matrix.png" width="50%">
</div>

#### Digression: SLAM和多视图几何中的「换系」和 「位姿（Pose）」

##### SLAM中的约定

- 换系：SLAM中，多以「换系」的视角来处理多个坐标系
- 参考坐标系：以坐标系$D$为参考坐标系是说：我们处在$D$坐标系下，所有点都表示为$D$下的坐标
- 位姿（Pose）：相机的位姿 = 位置$(x, y, z)$ + 姿态$(yaw, roll, pitch)$。在相机坐标系下谈论相机自身的Pose是没有意义的（因为相机就在原点，三个姿态角相对坐标轴均为零）。只有在非相机坐标系（此时该坐标系为参考系）下观察相机时Pose才是有意义的。如果将$(yaw, roll, pitch)$转换为旋转矩阵$R$，向量$t = (x, y, z)$为相机相对参考系原点的平移量，$T = \begin{pmatrix} R, t \\ 0^T, 1 \end{pmatrix}$称为相机的**位姿矩阵**。可以证明，$T$矩阵就是将坐标从相机坐标系换到参考坐标系（**camera-to-ref**）的变换矩阵

##### 证明：位姿矩阵就是换系（camera-to-ref）的变换矩阵

**Prerequisite**: [线性变换的几何意义](#数学基础理解线性变换的几何意义)

线性变换矩阵$A$的核心点

1. 换系：从目标系$D$观察任意给定系$S$，$S$的基向量在$D$下的坐标就是矩阵$A$的列向量，线性变换$P_D = AP_S$将任意给定点$P$在$S$下的坐标变为$D$下的坐标
2. 坐标/向量/刚体变换：通过对基向量的变换完备描述

- 换系视角：位姿矩阵就是换系的变换矩阵

在参考坐标系$D$（目标系，当前视角）下观察相机坐标系$S$的三个轴。整个坐标变换过程涉及：平移 + 线性变换下的换系，后者要求两个系的坐标原点必须重合。因此，先平移后换系更直观

<div align=center>
<img src="E:/weapons/Graphics/src/games101/MVP%20Transform/slam_coord_sys_transform.png" width="50%">
</div>

**我们需要一个中间坐标系$S^{\prime}$来处理平移**：$S^{\prime}$的原点与$D$重合，基向量与$S$共线且方向相同。需要这个中间系的原因是：只有当两个系的轴平行时，同一个向量在这两个系中的坐标才是相同的。用三角形法则处理平移，将点的坐标从$S$转到$S^{\prime}$下

$$P_{S^{\prime}} = P_S + t$$

$S^{\prime} \rightarrow D$的换系是一个线性变换（旋转）：找到$S^{\prime}$的基向量在$D$下的坐标表示$i^{\prime}, j^{\prime}, k^{\prime}$即得变换矩阵$R=(i^{\prime}, j^{\prime}, k^{\prime})$

$$ P_D = RP_{S^{\prime}} = RP_S + Rt$$

现在是关键，有三件重要的事：

a. 用「坐标变换」的观点看，**变换矩阵$R$描述了参考系$D$的基向量$i, j, k$到中间系$S^{\prime}$的基向量$i^{\prime}, j^{\prime}, k^{\prime}$的旋转**<br>
b. $i, j, k$到$i^{\prime}, j^{\prime}, k^{\prime}$的旋转是三维空间中的旋转，可以通过三次绕轴旋转描述<br>
c. 相机坐标系$S$的基向量与中间坐标系$S^{\prime}$的基向量相同（$i^{\prime}, j^{\prime}, k^{\prime}$，这是因为仅平移坐标系不改变向量的坐标）

因此，通过标定得到的$yaw, roll, pitch$这三个姿态角就是绕轴旋转的旋转角，旋转矩阵$R$完全由姿态角决定。如果$S^{\prime}$的基向量在$D$下的坐标表示较难直接获得，那就可以曲线救国：将$yaw, roll, pitch$这三个欧拉角转换为四元数，再转换为旋转矩阵即可得到$R$。这说明相机的位姿（Pose）与换系的变换矩阵完全等价。

一般情况下，**当提及相机的Pose时，默认说的是从相机坐标系换到参考坐标系（camera-to-ref）的变换矩阵**，对应变换方程

$$ P_{ref} = T_{rc}P_{cam}$$

##### MV变换与Pose的关系

- Pose是在当前参考坐标系下观察相机的位置和姿态。位姿矩阵$T$描述的是从相机坐标系转换到参考坐标系的**换系**矩阵
- ModelView变换是将相机（View）和场景（Model）作为一个整体，**平移+线性变换**到参考坐标系（标准位置）。变换前，参考坐标系下的相机和场景中物体的坐标是已知的。变换后，相机坐标系与参考系重合。因此相当于从参考系换到相机坐标系

从坐标变换的角度：

- MV变换：直接描述了坐标变换。相机（View）和场景（Model）同时向参考坐标系（标准位置）**平移 + 线性变换**，使变换后的相机坐标系与参考系重合
- Pose：用线性变换的几何意义将Pose换系转换为坐标变换：**参考坐标系（中的所有点）做线性变换（旋转到中间系$S^{\prime}$），然后平移到相机坐标系$S$**

从换系的角度：

- MV变换：从结果上看，相当于进行了换系，从参考坐标系换到相机坐标系。原先已知的是任意点在参考坐标系下的坐标。MV变换后，相机坐标系与参考坐标系重合，所有点的坐标处于相机坐标系下
- Pose就是在换系：从相机坐标系换到参考坐标系

#### Projection Transformation: prepare for projection

投影的目的：划定一定范围内的model（这个范围又称为裁剪空间），将远平面到近平面所有的内容投影到近平面上

注意：本节中的所有投影变换本身并未做投影这件事（投影是要降维的），而是**为投影做准备**：将model变换到标准立方体$[-1, 1]^3$内

<div align=center>
<img src="E:/weapons/Graphics/src/games101/MVP%20Transform/projection_range_[near,%20far].png" width="50%">
</div>

- 为什么没有drop $z$？

有两个主要原因：

1. 次要原因：直观上，如果场景中只有一个物体，当然可以drop $z$完成投影，但实际场景不止一个物体，在相机视角下都会有遮挡关系。
2. 主要原因：场景物体是连续的，屏幕是离散的，这两者间需要一步离散化的操作，这就是渲染中的光栅化，在下面一大章完成这件事

##### 正交投影与透视投影的关系

相机无穷远，透视投影与正交投影相同：相机离近平面（和远平面）越远，近、远两平面的尺寸差距越小，光线平行度越高。如果相机（距离近平面）无穷远，且近平面和远平面距离有限，那么两平面大小相同，光线平行，透视投影变为正交投影。

#### Orthographic projection

简单理解：相机处于标准位置，扔掉$z$坐标，然后将结果rescale到$[-1, 1]^2$范围内（长宽缩放比可能不一样，aspect ratio会变）

标准做法：相机在标准位置，model包含在$[l, r] \times [b, t] \times [f, n]$的立方体内。将该立方体中心平移到原点，然后scale到$[-1, 1]^3$。变换矩阵记为$M_{ortho}$

$$ M_{ortho} = \underbrace{\begin{pmatrix}
\frac{2}{r - l} & 0 & 0 & 0 \\
0 & \frac{2}{t - b} & 0 & 0 \\
0 & 0 & \frac{2}{n - f} & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}}_{scale} \underbrace{\begin{pmatrix}
1 & 0 & 0 & -\frac{r + l}{2} \\
0 & 1 & 0 & -\frac{t + b}{2} \\
0 & 0 & 1 & -\frac{n + f}{2} \\
0 & 0 & 0 & 1
\end{pmatrix}}_{translation} = \begin{pmatrix}
\frac{2}{r - l} & 0 & 0 & \frac{r + l}{r-l} \\
0 & \frac{2}{t - b} & 0 & \frac{b + t}{b-t} \\
0 & 0 & \frac{2}{n - f} & \frac{f + n}{f-n} \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

注：相机看向$-z$方向，$n > f$

#### Perspective Projection

对于透视投影的视锥，如果能将远平面变得跟近平面一样大，那么再利用正交投影就可完成整个透视投影。因此透视投影分两步走：

- 将frustum变换为正交投影的cuboid
- 正交投影转换到$[-1, 1]^3$

##### frustum -> orthographic cuboid

视锥内任一点$(x, y, z)$，投影到近平面上的位置容易找到：相似三角形即可

<div align=center>
<img src="E:/weapons/Graphics/src/games101/MVP%20Transform/persepective_projection_similar_triangle.png" width="50%">
</div>

$$
x' = \frac{n}{z} x, \hspace{2pt} y' = \frac{n}{z} y
$$

因此目标是寻找映射关系

$$ \begin{pmatrix}
x \\ y \\ z \\ 1
\end{pmatrix} \rightarrow
\begin{pmatrix}
x^{\prime} \\ y^{\prime} \\ z^{\prime} \\ 1
\end{pmatrix} = \begin{pmatrix}
nx/z \\ ny/z \\ unknown \\ 1
\end{pmatrix}
$$

如果直接将上式转化为矩阵形式

$$ \begin{pmatrix}
x' \\ y' \\ z' = ? \\ 1
\end{pmatrix}
= \begin{pmatrix}
n/z & 0 & 0 & 0 \\
0 & n/z & 0 & 0 \\
? & ? & ? & ? \\
0 & 0 & 0 & 1
\end{pmatrix} \begin{pmatrix}
x \\ y \\ z \\ 1
\end{pmatrix}
$$

由于目前还不知道$z \rightarrow z^{\prime}$的映射关系，因此变换矩阵的第三行元素、输出$z^{\prime}$暂时用问号代替。上式中，变换矩阵与输入点的$z$坐标相关，这是有问题的：变换矩阵描述的是一般变化，应与输入无关。造成这个问题的主要原因是$1/z$的系数。

任意齐次坐标乘以任意不为0的常数，表示的点不变。那么就将矩阵中的$1/z$提出来，乘到等式左边的输出坐标即可

$$ \begin{gather}
\begin{pmatrix}
x \\ y \\ z \\ 1
\end{pmatrix} \rightarrow
\begin{pmatrix}
x^{\prime} \\ y^{\prime} \\ z^{\prime} \\ 1
\end{pmatrix} = \begin{pmatrix}
nx/z \\ ny/z \\ unknown \\ 1
\end{pmatrix} \overset{\times z}{==} \begin{pmatrix}
nx \\ ny \\ still \hspace{3pt} unknown \\ z
\end{pmatrix} \\ \\
\begin{pmatrix}
nx \\ ny \\ ? \\ z
\end{pmatrix}
= \begin{pmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
? & ? & ? & ? \\
0 & 0 & 1 & 0
\end{pmatrix} \begin{pmatrix}
x \\ y \\ z \\ 1
\end{pmatrix}
\end{gather}
$$

注意两件事：

- 变换的直接输出是$(nx, ny, ?, z)$，最后一维$w$是输入点的$z$
- 变换矩阵的最后一行在计算上有无数组解：$(0, 0, 1, 0) \hspace{3pt} or \hspace{3pt} (0 ,0, k, (1-k)z), \hspace{2pt} k\in R$，后一组解再次与输入$z$有关，舍弃

将系数放到输出有三个原因：

- 计算方便：输入不用处理，输出直接用最后一维$w=z$归一化即可
- 归一化操作不影响后续其他线性变换
- 更high level的原因：透视除法。透视反映了近大远小的关系：$x \rightarrow nx / z \Rightarrow \Delta x \rightarrow n\Delta x / z$

现在只差最后一步：找到映射关系$z \rightarrow z^{\prime}$，确定矩阵的第三行。将frustum变换为正交投影的cuboid是一种“挤压”的操作：

- 近平面无需挤压，因此任意近平面上的点都不发生移动：$(x, y, n, 1) \rightarrow (x, y, n, 1) == (nx, ny, n^2, n)$（注意$w == z$）
- 远平面挤压后，所有点的$z$不发生改变：$z = f \rightarrow z^{\prime} = f$
- 以$(0, 0, f, 1)$为中心挤压远平面。这意味着挤压后中心不变：$(0, 0, f, 1) \rightarrow (0, 0, f^2, f)$（注意$w == z$）

限定以上三条规则后，挤压方法唯一，根据待定系数确定矩阵的第三行

$$ \begin{gather}
\begin{pmatrix}
? & ? & ? & ?
\end{pmatrix} \begin{pmatrix}
x \\ y \\ n \\ 1
\end{pmatrix} = n^2 \Rightarrow
\begin{cases}
\begin{pmatrix}
? & ? & ? & ?
\end{pmatrix} = \begin{pmatrix}
0 & 0 & A & B
\end{pmatrix} \\ \\
An + B = n^2
\end{cases} \\
\begin{pmatrix}
0 & 0 & A & B
\end{pmatrix} \begin{pmatrix}
0 \\ 0 \\ f \\ 1
\end{pmatrix} = f^2 \Rightarrow
Af + B = f^2
\end{gather}
$$

解得$A = n + f, \hspace{2pt} B = -nf$，变换矩阵即为

$$ M_{persp\rightarrow ortho} = \begin{pmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
0 & 0 & n + f & -nf \\
0 & 0 & 1 & 0
\end{pmatrix}
$$

##### perspective projection

将变为cuboid的视锥进行正交投影，就是完整的透视投影

$$ M_{persp} = M_{ortho}M_{persp \rightarrow ortho} = \begin{pmatrix}
\frac{2}{r - l} & 0 & 0 & \frac{r + l}{r-l} \\
0 & \frac{2}{t - b} & 0 & \frac{b + t}{b-t} \\
0 & 0 & \frac{2}{n - f} & \frac{f + n}{f-n} \\
0 & 0 & 0 & 1
\end{pmatrix} \begin{pmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
0 & 0 & n + f & -nf \\
0 & 0 & 1 & 0
\end{pmatrix}
$$

#### MVP Transform

到目前为止，我们已经完成了MVP变换中的所有components，将视锥 or orthographic cuboid转化为$[-1, 1]^3$内的标准的cuboid，这个cuboid简称为NDC（标准化设备坐标，Normalized Device Coordinate）。在变换部分，还有最后的一个视口变换（viewport transformation）将NDC在$x, y$方向上拉伸为图像的$width$和$height$，便于成像（渲染）。

在解决视口变换前，我们需要把整个过程快速复现一下，对MVP变换有一个宏观的认识。

MVP变换过程：MV变换（从任意参考系换到相机坐标系下）-> 透视投影（挤压frustum）-> 正交投影

$$ M_{MVP} = M_{ortho}M_{persp \rightarrow ortho}M_{view} $$

#### Viewport Transform

先定义屏幕（screen）：像素定义为内部颜色不会发生变化的小方块（pixel square）

<div align=center>
<img src="E:/weapons/Graphics/src/games101/screen_def.png" width="50%">
</div>

注意：

- 图像是场景内物体在屏幕上渲染的结果，渲染前的空间仍然是连续的，因此**屏幕仍然处于连续空间中**，每个pixel都是**连续的square**
- pixel **square**（而非pixel中心）的坐标$x$和$y$是离散的整数。也就是说，每个像素的长宽均为1
- pixel中心在square的中心，坐标为$(x + 0.5, y + 0.5)$
- 整个屏幕覆盖所有pixel square，因此屏幕处于从$(0, 0)$到$(width, height)$的连续区域内

> Note：像素中心坐标的定义在不同的教材中会略有差别，但其他核心定义不会变
> 
> e.g. 虎书将像素中心定义在整数坐标上
> 
> <img src="E:/weapons/Graphics/src/games101/screen_def_tiger_book.png" width="50%">

视口变换非常简单：将NDC从$[-1, 1]^3$变到$x \in [0, width] \times y \in [0, height] \times z\in [-1, 1]$，也就是说$z$不变，在$x, y$上平移和缩放<br>
（忽略$z$的$(x, y)$在三维空间中表示一条平行于$z$轴的直线）

不难写出视口变换矩阵

$$ M_{vp} = \underbrace{\begin{pmatrix}
width / 2 & 0 & 0 & 0 \\
0 & height / 2 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}}_{rescale} \underbrace{\begin{pmatrix}
1 & 0 & 0 & 1 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}}_{translation} = \begin{pmatrix}
width / 2 & 0 & 0 & width / 2 \\
0 & height / 2 & 0 & height / 2 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

（这里先缩放后平移，因此需要注意平移的offset）

至此，变换结束

$$ M = M_{vp}M_{ortho}M_{persp \rightarrow ortho}M_{view} $$

--------------------

### Rendering

MVP + 视口变换将空间中的物体（model in a frustum or cuboid）变到了$x \in [0, width] \times y \in [0, height] \times z \in [-1, 1]$的范围内。接下来就是要把东西画在屏幕上，这就是**渲染（Rendering）**

### Rasterization

场景物体表面可以分解为若干多边形，图形学使用三角形作为最基本的多边形

- why triangle？

（1）三角形是最简单的多边形，任何多边形都可以打散为若干个三角形的组合<br>
（2）三角形内部必是平面：四边形沿着对角线稍微一折就变成了非平面的<br>
（3）三角形内部和外部是清晰的：不会有空洞，不会有凹凸性的问题

光栅化（Rasterization）阶段**计算多边形对像素点的覆盖**，不考虑着色问题

#### Rasterization as 2D sampling

1. 判断给定点在三角形内部 or 外部

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/point_in_triangle.png" width="30%"> <img src="E:/weapons/Graphics/src/games101/rendering/point_out_triangle.png" width="30%">
</div>

定一个三角形的环绕方向 -> **叉积结果的$z$分量同号则在内部，反之则在外部**。

例如上左图，$\overrightarrow{AB} \times \overrightarrow{AP}, \overrightarrow{BC} \times \overrightarrow{BP}, \overrightarrow{CA} \times \overrightarrow{CP}$的$z$分量均为正，则$P$点在$\triangle ABC$内。右图，$\overrightarrow{P_2P_0} \times \overrightarrow{P_2Q}$的$z$为负，而$\overrightarrow{P_0P_1} \times \overrightarrow{P_0Q}$和$\overrightarrow{P_1P_2} \times \overrightarrow{P_1Q}$的$z$为正，三者不同号，因此$Q$点在$\triangle P_0P_1P_2$外。

Corner case: 点落在三角形边上，图形学里不做统一定义，自行规定处理方式即可。

在实际光栅化计算时，三角形顶点为三维向量，像素中心坐标为二维向量，计算叉积时只需要drop $z$ or 赋任意值，因为叉积结果的$z$与输入向量的$z$无关。Here's the proof

$$ \vec{a} \times \vec{b} = \begin{vmatrix}
\vec{i} & \vec{j} & \vec{k} \\
x_1 & y_1 & z_1 \\
x_2 & y_2 & z_2
\end{vmatrix}
$$

判断像素中心是否在三角形内部，只需判断两个向量叉积结果的$z$坐标是否同号即可，因此对上式按第一行展开，只取$\vec{k}$项

$$ z = (x_1 y_2 - x_2 y_1) \vec{k} $$

无论从结果还是展开过程上看，$z$坐标与原始向量$\vec{a}, \vec{b}$的$z_1, z_2$无关，因此可以给$z_1, z_2$赋任意值，或更简单，直接根据上式计算$x_1 y_2 - x_2 y_1$，判断符号即可。

2. 采样（sampling）是最简单的光栅化方式

基本规则：若pixel中心在三角形内，则pixel square属于三角形。有三种遍历像素的方法：

- 遍历整图
- 遍历最小外接正矩形（bbox）：显然完整遍历整张图是低效的。一种加速方法是找到三角形的最小外接正矩形，遍历这个bbox内的每个pixel即可
- 增量遍历：如果是一个扁长的三角形，遍历bbox效率同样不高。最理想情况是一个像素都不多考虑，但这件事做起来不容易，暂且按下不表

简单采样做光栅化的问题很明显，视觉上不自然，出现锯齿（Jaggies），走样（Aliasing）

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/aliasing_example_0.png" width="30%"> <img src="E:/weapons/Graphics/src/games101/rendering/aliasing_example_1.png" width="30%">
</div>

#### Antialiasing

1. 走样的原因

场景是一个三维空间上的连续函数，包含几何覆盖关系、着色参数和着色方程。为了将场景显示在屏幕上（渲染），需要将场景离散化到一个个pixel上，这个过程导致了锯齿。

采样是直接且普遍使用的离散化方法。根据奈奎斯特采样定理，只有采样率大于等于信号最高频的两倍时，才能通过离散的采样点完美恢复原始信号。当信号变化快（有较多的高频分量）而采样率不足时就会出现走样，因此走样在频域上的解释就是频谱混叠。这个理论同时给出了两个解决方案：（1）首选增大采样率（2）如果提高采样率的开销过大，就要先滤掉出现混叠的高频分量

2. SuperSampling Anti-Aliasing (SSAA)

SSAA最简单粗暴，直接增大采样率来解决问题。假设屏幕输出分辨率为$H \times W$，$n^2 \times$SSAA首先渲染一个$nH \times nW$的buffer，然后下采样到$H \times W$。具体来说：

**step1** 每个pixel square内采样$n$个点，采样点分布方式不限，统称为采样模板<br>
**step2** 计算像素中心是否被三角形覆盖<br>
**step3** run pixel shader进行着色<br>
**step4** 下采样渲染结果到$H \times W$

注意，光栅化只涉及采样和计算覆盖的过程，不包含第三、四两步。

**Preview**：超采样直接渲染出来的是一个$n$倍于target size的图像，最终需要下采样到屏幕分辨率才可以显示，这个过程称为**resolve**。对于SSAA，resolve就是下采样，而对于接下来的MSAA，resolve相当于均值滤波。

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/SSAA_supersampling.png" width="50%">
</div>

SSAA在效果上是最好的抗锯齿方法，代价就是$n^2$的计算复杂度。光栅化计算量较低，这个代价可以接受。但着色阶段的计算开销大，需要优化这个开销。

3. Multi-Sampling Anti-Aliasing (MSAA)

既然着色的开销大，那就仅在光栅化阶段使用supersampling而不对子采样点着色。MSAA在光栅化阶段接受$n^2$的supersampling，与SSAA相同。不同点在于，MSAA计算每个pixel supersampling的覆盖率，而不直接对每个子采样点着色。在着色阶段，对于覆盖率大于0的pixel运行一次pixel shader，并将颜色乘以覆盖率。 

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/MSAA_average.png" width="50%">
</div>

理论上，MSAA的resolve实际是在连续的三角形上做均值滤波，卷积核大小等于一个pixel square的大小。卷积中的积分运算并未使用解析解，而是用离散采样求和的方式实现。

#### 图形学与机器学习中的积分：闭式解，上下界以及离散求和近似

机器学习中，如果在数学建模或优化目标中出现了积分式，倾向于利用数学方法求出其闭式解或寻找上下界，以便于进行优化，例如GAN的理论求解。

图形学处理的三维空间中的几何形状的位置、状态多变。以三角形的反走样为例，其边界直线的位置、长度有无数种可能，故而被积函数及积分的上下界就用无数种可能，求一个具有一般性的解析解比较困难。因此**图形学中更倾向于用离散求和去近似积分运算**。另一方面，离散求和的好处是可以最大程度地利用GPU的并行计算能力进行加速，在硬件层面上达到更高的处理效率。

解析求解与采样并无绝对意义上的优劣，二者并不矛盾。闭式解难求时就用采样求和进行数值近似，闭式解可求时就考虑显式优化。

#### z-buffer (深度缓冲、深度测试)

梳理一下光栅化过程：MVP+视口变换得到$x \in [0, width] \times y \in [0, height] \times z\in [-1, 1]$的立方体 -> 屏幕空间像素采样 + 反走样的超采样部分 -> 判断pixel center是否inside triangle，确定像素覆盖。

为了将场景画在屏幕上，接下来还有两件事：1. 处理遮挡关系（可见性）2. 着色。z-buffer用于处理遮挡关系。

相机视角下，三维场景中的物体由于在相机坐标系下的深度不同，可能会出现遮挡（occlusion）。z-buffer就是进行深度判断，确定渲染后物体在图像上的**可见性（visibility**）。

算法本身非常朴素：遍历每个三角形，记录所有**采样点**的最小深度

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/z-buffer.png" width="50%">
</div>

### Shading

光栅化完成后，我们确定了场景内物体对屏幕上像素的覆盖关系，做了抗锯齿，也处理了相机视角下物体的遮挡关系（可见性）。接下来将物体颜色直接分配给对应的像素似乎就完成了渲染。但除了颜色，明暗的不同也会影响物体颜色的观测结果。所以要同时引入物体颜色、明暗等因素构建数学模型来描述着色过程。

图形学对着色（shading）的定义：在物体上应用材质（material）的过程，或者说是根据物体材质进行染色的过程。不同的**材质**与**光线**相互作用，产生不同的视觉效果。

#### shading is local

着色模型只考虑着色点附近的很小的一块区域，因此这个范围内的物体表面（surface）可以视为一个平面。与之相对的，对物体在地面上的阴影着色就由non-local的模型负责。

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/shading_shadow.png" width="50%">
</div>

> **local & non-local**: local和non-local是相对的，**超出local model建模的区域就是non-local的**。例如，shading建模光照时只考虑着色点附近的一小块区域，因此shading is local。阴影（shadow）属于shading建模区域外的部分，因此对shading来说，shadow就是non-local的。
> 
> 理解why model is local的时候，举出non-local的例子，对理解会有很大帮助。

#### shading input

着色的建模过程集中在着色点（shading point）上，在着色点附近的极小范围内的物体表面可视为平面。建立光照模型需要：

- 表面法向$n$
- 光照方向$l$
- 相机视角方向$v$
- 物体表面（材质）参数，例如颜色，光泽度等

$n, l, v$均为单位向量

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/shading_input.png" width="50%">
</div>

> 颜色：表面对不同波长 or 频率的光的吸收率，是$\lambda$ or $f$的函数
> 光泽度（gloss/shininess）：表面在镜面反射方向上的反射能力

#### Blinn-Phong Reflectance Model

直观感受上，一个真实场景成像的光照主要分为三部分

- 镜面高光：观察视角处于光线的镜面反射附近，形成高光
- 漫反射：物体表面不平整，光线向各个方向均匀反射
- 环境光（间接光照）：物体不接受直接光照的位置也能看到颜色，是反射了其他物体的反射光

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/blinn-phong_light_type.png" width="50%">
</div>

Blinn-Phong是基础的光线反射模型，主要建模镜面高光和漫反射，最复杂的环境光部分用常系数简单处理，最终的模型是这三部分的简单求和。

##### point light

一般将光源视为点光源，光的传播面是一个球面，球面上光强均匀分布。光源功率一定，在球面上的能量（光强在球上的面积分）就是固定的，由此可推出任意距光源$r$处的光照强度$I_r$

$$I \cdot 4\pi = I_r \cdot 4 \pi r^2 \Rightarrow I_r = I / r^2$$

$I$为单位距离$r=1$处的光强

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/point_light_intensity.png" width="50%">
</div>

##### Diffusion reflection

- 漫反射：光线入射到粗糙的物体表面，反射光向各个方向射出，而不仅仅是镜面反射方向
- 朗伯反射（Lambertian reflectance）：物体表面是理想的漫反射表面，反射光的光强在反射面上均匀分布。Blinn-Phong使用该模型建模漫反射

- Lambert's cosine law

除光源外，物体表面接受的入射光总量也会影响反射光强。正式的说法是：当光源强度一定时，物体表面漫反射的反射光强与入射光强正相关。Lambert's law证明，单位面积接收的光强正比于入射角的余弦

$$I_{rec} \propto \cos \theta = n \cdot l$$

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/blinn-phong_diffusion_lambert_law.png" width="50%">
</div>

- Lambertian diffuse shading

反射光需要同时考虑入射光和物体表面材质对不同频率的光的吸收率，并且光源只能出现在反射面的上半部分（$0\degree \leq$ 入射角 $< 90\degree$）

$$L_d = k_d (I / r^2) \max(0, n \cdot l)$$

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/blinn-phong_lambertian_diffuse_shading.png" width="50%">
</div>

##### Specular reflection

看到高光的强弱程度取决于观测方向是否接近于反射光的方向。这里有两种方法：Phong模型直接计算镜面反射方向，Blinn-Phong模型计算半程向量（half vector）。

- 半程向量（half vector）

直观上我们希望计算观测方向与镜面反射方向的接近程度。镜面反射方向不是直接已知量，但法向是光照方向与镜面反射方向的角分线方向。因此自然想到比较「表面法向」与「光照方向与观测方向的角分线方向」的接近程度

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/blinn-phong_specular_half_vector.png" width="50%">
</div>

- 镜面反射方向

直接求解镜面反射方向难度并不大：设沿镜面反射方向的单位向量为$r$。$l + r$的方向与$n$相同，$l, r$长度相等，那么$l, r, k*n \hspace{3pt} (k \in R)$就能构成一个等腰三角形

$$2(l \cdot n)n = l + r \Rightarrow r = 2(l \cdot n)n - l$$

可以验证这个结果的正确性：入射角与反射角相等，因此应有$r\cdot n$等于$l\cdot n$，简单计算$r\cdot n$即可验证等式成立。

将Blinn-Phong镜面反射的$n \cdot h$替换为$v \cdot r$即得Phong模型的镜面反射项

$$L_s = k_s(I / r^2)\max (0, v\cdot r)^p$$

##### Ambient light

环境光是非常复杂的弹弹乐，Blinn-Phong简单将这一项处理为常数项

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/blinn-phong_ambient_term.png" width="50%">
</div>

##### Blinn-Phong reflectance model

完整的Blinn-Phong光照模型就是以上三部分求和

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/blinn-phong_reflectance_model.png" width="50%">
</div>

