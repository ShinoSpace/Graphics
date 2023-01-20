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

先定义屏幕（screen）：就是通常的图像定义。到目前为止，一个像素就是一个内部颜色不会发生变化的小方块（square）。这种定义方式不严谨，因为像素本身是比较复杂的东西，但现在这样理解没有问题（之后再做修正）。

屏幕（图像）空间

<div align=center>
<img src="E:/weapons/Graphics/src/games101/screen_def.png" width="50%">
</div>

注意：

- pixel **square**（而非pixel中心）的坐标$x$和$y$是离散的整数
- pixel中心在square的中心，坐标为$(x + 0.5, y + 0.5)$，因此每个pixel square占据$1 \times 1$的空间
- 整个屏幕覆盖所有pixel square，因此屏幕的范围是$(0, 0)$到$(width, height)$

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

MVP + 视口变换将空间中的物体（model in a frustum or cuboid）变到了$x \in [0, width] \times y \in [0, height] \times z \in [-1, 1]$的范围内，接下来就是要把东西画在屏幕上

#### Rasterization

物体表面可以分解为若干多边形，光栅化（Rasterization）将这些多边形打散为像素点，显示在屏幕上。那么就需要每个多边形告诉对应像素的RGB值都是多少。

- why triangles?

（1）三角形是最简单的多边形，任何多边形都可以打散为若干个三角形的组合<br>
（2）三角形内部必是平面：四边形沿着对角线稍微一折就变成了非平面的<br>
（3）三角形内部和外部是清晰的：不会有空洞，不会有凹凸性的问题

##### Rasterization as 2D sampling

问题：如何在屏幕上表示给定三角形？也就是说如何将三角形打散为像素？

1. 判断一个点在三角形内部和外部

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/point_in_triangle.png" width="30%"> <img src="E:/weapons/Graphics/src/games101/rendering/point_out_triangle.png" width="30%">
</div>

定一个三角形的环绕方向 -> 外积向量方向一致则在内部，反之则在外部。左图，$\overrightarrow{AB} \times \overrightarrow{AP}, \overrightarrow{BC} \times \overrightarrow{BP}, \overrightarrow{CA} \times \overrightarrow{CP}$同向，$P$在$\triangle ABC$内。右图同理，$Q$在$\triangle P_0P_1P_2$外。

Corner case：点落在三角形边上，图形学里不做统一定义，自行规定处理方式即可。

2. 采样（sampling）是最简单的光栅化方式

若pixel中心在三角形内，则pixel square属于三角形

- 遍历整图

- 遍历最小外接正矩形（bbox）：显然完整遍历整张图是低效的。一种加速方法是找到三角形的最小外接正矩形，遍历这个bbox内的每个pixel即可

- 增量遍历：如果是一个扁长的三角形，遍历bbox效率同样不高。最理想情况是一个像素都不多考虑，但这件事做起来不容易，暂且按下不表

简单采样做光栅化的问题很明显，视觉上不自然，出现锯齿（Jaggies），走样（Aliasing）

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/aliasing_example_0.png" width="30%"> <img src="E:/weapons/Graphics/src/games101/rendering/aliasing_example_1.png" width="30%">
</div>

##### Antialiasing: method

理论比较长。先理解怎么做，然后从理论上解释为什么这样做

1. 如何做反走样

  （1）直观上，走样源于采样不足，所以最简单的做法就是增大采样率。在图像/屏幕空间下就是（屏幕尺寸保持不变）增多像素的个数（换了一个分辨率更高的屏幕）<br>
  （2）理论上，走样的原因是采样频率低导致频域上频谱混叠。解决方法是先将高于采样率的频谱过滤掉，即低通滤波（模糊），然后采样

2. MSAA

直观上，当屏幕固定时，采样率低导致pixel center比较稀疏，每个pixel square的size较大，那么简单采样时就会丢掉很多near bound, but out of bound的像素。考虑到屏幕本身不能改变，那么就需要采用更加soft的分配方式。MSAA在单个像素内subsample多个位置，然后为这些亚像素分配灰度，最终每个像素的灰度就是亚像素的平均

<div align=center>
<img src="E:/weapons/Graphics/src/games101/rendering/MSAA_0_subsample.png" width="30%"> <img src="E:/weapons/Graphics/src/games101/rendering/MSAA_1_average_0.png" width="30%"> <img src="E:/weapons/Graphics/src/games101/rendering/MSAA_1_average_1.png" width="30%"> <img src="E:/weapons/Graphics/src/games101/rendering/MSAA_2_result.png" width="30%">
</div>

MSAA的本质是在连续的三角形上做均值滤波，卷积核大小等于一个pixel square的大小。卷积中的积分运算并未使用解析解，而是用离散采样求和的方式实现

##### 图形学与机器学习中的积分：闭式解，上下界以及离散求和近似

机器学习中，如果在数学建模或优化目标中出现了积分式，倾向于利用数学方法求出其闭式解或寻找上下界，例如GAN的理论求解。而图形学处理的三维空间中的几何形状的位置、状态多变。以三角形的反走样为例，其边界直线的位置、长度有无数种可能，故而被积函数及积分的上下界就用无数种可能，求一个具有一般性的解析解比较困难。因此**图形学中更倾向于用离散求和去近似积分运算**。另一方面，离散求和的好处是可以最大程度地利用GPU的并行计算能力进行加速，在硬件层面上达到更高的处理效率

##### 理论是为算法打底的

还是以反走样为例，MSAA的理论根基是傅里叶变换、低通滤波及采样定理，最终的实现进行了工程化处理，用离散求和去近似积分操作（连续求和），以求在效果和开销之间trade off

因此，理论是为算法打了一个底，保证了算法一定是对的