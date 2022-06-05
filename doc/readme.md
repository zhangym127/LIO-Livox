
# IMU预积分总结与公式推导

本文是对【泡泡机器人】刊载的邱笑晨编写的《IMU预积分总结与公式推导》的理解和整理，基本上和原文一致，个别地方根据本人习惯对术语等做了调整，例如理想值改为真值。

## 1、概述
在基于BA的视觉惯性融合算法中，各个节点的载体（车辆）状态都是有待优化的量。IMU预积分的初衷，是希望借鉴纯视觉SLAM中图优化的思想，将帧与帧之间IMU相对测量信息转换为约束节点（载体位姿）的边参与到优化框架中。

### 1.1 IMU预积分测量值

IMU预积分不仅仅是一个理论，而且是一个实践指南，它给出了基于IMU预积分进行位姿优化的具体方法。优化的核心是构造代价函数，而代价函数的核心是构造残差，IMU预积分给出了具体的残差定义（详见本文第七章），即相邻两帧（点云）之间位姿增量的估计值和测量值之差：
$$
残差_{ij}=位姿增量估计值_{ij}-位姿增量测量值_{ij}
$$
其中估计值通常需要通过非IMU的方式获得，例如通过点云匹配，测量值就来自IMU预积分。在图优化的过程中，要进行局部甚至全局的反复优化，随着优化的推进，每个节点的位姿和噪声都会发生变化，残差中的位姿增量测量值就需要重新计算，IMU预积分就提供了一个近似的测量值修正方法，免去了积分的重新计算，是预积分降低计算量的关键。这个残差公式可以直接添加到诸如Ceres的优化框架中，用以实现一个代价函数。例如LIO-Livox就使用了与本文第七章完全一致的残差来构造代价函数。

因此本文档自始至终的一个主线就是求解相邻两帧之间IMU预积分测量值，试图通过在既有IMU预积分测量值上添加一个近似修正量的方式来避免重新积分。

#### 1.2 IMU预积分测量值的修正量

#### 1.3 IMU预积分测量噪声协方差和优化权重

#### 1.4 重力加速度

传统的SLAM算法很少使用加速度来计算速度和位置，原因是加速度计的测量值是包含反向重力的比力，而不是纯加速度。这使得一旦姿态不准确，重力投影误差将对速度和位置积分产生严重影响。

IMU预积分理论最大的贡献是对这些IMU相对测量进行处理，使得它与绝对位姿解耦（或者只需要线性运算就可以进行校正），从而大大提高优化速度。另外，这种优化架构还使得加计测量中不受待见的重力变成一个有利条件——重力的存在将使整个系统对绝对姿态（指相对水平地理坐标系的俯仰角和横滚角，不包括真航向）可观。要知道纯视觉VO或者SLAM是完全无法得到绝对姿态的。

IMU预积分理论直接使用加速度的测量值进行速度和位置的估计，没有使用诸如编码器、毫米波雷达等测速装置。

### 2、预积分的引出
假设$k=i$帧的姿态、速度、位置分别是$R_i$、$v_i$、$p_i$，则可以利用从$k=i$到$k=j-1$帧的所有IMU测量，直接更新得到$k=j$帧的$R_j$、$v_j$、$p_j$，详细如下：
$$
\mathbf{R}_{j}=\mathbf{R}_{i} \cdot \prod_{k=i}^{j-1} \operatorname{Exp}\left(\left(\tilde{\boldsymbol{\omega}}_{k}-\mathbf{b}_{k}^{g}-\boldsymbol{\eta}_{k}^{g d}\right) \cdot \Delta t\right)\tag{1}
$$
其中，$\tilde{\boldsymbol{\omega}}_{k}$是第$k$帧的角速度的测量值，$\mathbf{b}_{k}^{g}$是第$k$帧的角速度偏差，$\boldsymbol{\eta}_{k}^{g d}$是第$k$帧的角速度测量噪声。
$$
\mathbf{v}_{j}=\mathbf{v}_{i}+\mathbf{g} \cdot \Delta t_{i j}+\sum_{k=i}^{j-1} \mathbf{R}_{k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{k}^{a}-\mathbf{\eta}_{k}^{a d}\right) \cdot \Delta t\tag{2}
$$
其中，$\mathbf{g}$是重力加速度，$\tilde{\mathbf{f}}_{k}$是第$k$帧的加速度测量值，$\mathbf{b}_{k}^{a}$是第$k$帧的加速度偏差，$\boldsymbol{\eta}_{k}^{a d}$是第$k$帧的加速度测量噪声。
加速度的测量值耦合了重力加速度，因此需要加上一个$\mathbf{g} \cdot \Delta t_{i j}$进行抵消。

$$
\begin{aligned}
\mathbf{p}_{j} &=\mathbf{p}_{i}+\sum_{k=i}^{j-1}\left[\mathbf{v}_{k} \cdot \Delta t+\frac{1}{2} \mathbf{g} \cdot \Delta t^{2}+\frac{1}{2} \mathbf{R}_{k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{k}^{a}-\mathbf{\eta}_{k}^{a d}\right) \cdot \Delta t^{2}\right]
\end{aligned}\tag{3}
$$
<font color=Blue>加速度的测量值耦合了重力加速度，因此需要加上一个$\frac{1}{2}\mathbf{~g}\cdot\Delta t^{2}$进行抵消。</font>

为了避免每次更新初始的$R_i$、$v_i$、$p_i$都要重新积分求解$R_j$、$v_j$、$p_j$，引出**预积分真值**如下，这里应用了正交矩阵（旋转矩阵）的转置等于正交矩阵的逆的性质：
$$
\begin{aligned}
\Delta \mathbf{R}_{i j} & \triangleq \mathbf{R}_{i}^{T} \mathbf{R}_{j} \\
&=\prod_{k=i}^{j-1} \operatorname{Exp}\left(\left(\tilde{\boldsymbol{\omega}}_{k}-\mathbf{b}_{k}^{g}-\boldsymbol{\eta}_{k}^{g d}\right) \cdot \Delta t\right)
\end{aligned}\tag{4}
$$
$$
\begin{aligned}
\Delta \mathbf{v}_{i j} & \triangleq \mathbf{R}_{i}^{T}\left(\mathbf{v}_{j}-\mathbf{v}_{i}-\mathbf{g} \cdot \Delta t_{i j}\right) \\
&=\sum_{k=i}^{j-1} \Delta \mathbf{R}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{k}^{a}-\mathbf{\eta}_{k}^{a d}\right) \cdot \Delta t
\end{aligned}\tag{5}
$$
$$
\begin{aligned}
\Delta \mathbf{p}_{i j} & \triangleq \mathbf{R}_{i}^{T}\left(\mathbf{p}_{j}-\mathbf{p}_{i}-\mathbf{v}_{i} \cdot \Delta t_{i j}-\frac{1}{2} \mathbf{g} \cdot \Delta t_{i j}^{2}\right) \\
&=\sum_{k=i}^{j-1}\left[\Delta \mathbf{v}_{i k} \cdot \Delta t+\frac{1}{2} \Delta \mathbf{R}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{k}^{a}-\boldsymbol{\eta}_{k}^{a d}\right) \cdot \Delta t^{2}\right]
\end{aligned}\tag{6}
$$
<font color=Blue>注意：上面三个预积分公式中的$\Delta \mathbf{R}_{i j}$、$\Delta \mathbf{v}_{i j}$、$\Delta \mathbf{p}_{i j}$并不是i、j帧之间姿态、速度、位置真值的变化量，而是与i、j帧之间IMU预积分测量值对应的真值，由于IMU预积分测量值耦合了重力加速度，因此对应的IMU预积分真值也必须含有一个重力加速度的分量，否则无法解释速度的变化量为什么还要减去$\mathbf{g} \cdot \Delta t_{i j}$。</font>


### 3、测量值=真值+噪声
下面分别对$\Delta \mathbf{R}_{i j}$、$\Delta \mathbf{v}_{i j}$、$\Delta \mathbf{p}_{i j}$进行整理，尝试将噪声项（$\boldsymbol{\eta}_{k}^{g d}$和$\boldsymbol{\eta}_{k}^{a d}$）从预积分真值中分离出来，使预积分具有“测量值=真值+噪声”的形式。<font color=Blue>假设在预积分的区间内，两帧间的偏差是相等的，即$\mathbf{b}_{i}^{g}=\mathbf{b}_{i+1}^{g}=\cdots=\mathbf{b}_{j}^{g}$以及$\mathbf{b}_{i}^{a}=\mathbf{b}_{i+1}^{a}=\cdots=\mathbf{b}_{j}^{a}$。</font>

#### 3.1 $\Delta \mathbf{R}_{i j}$
对于$\Delta \mathbf{R}_{i j}$则有：
$$
\begin{aligned}
\Delta \mathbf{R}_{i j} &=\prod_{k=i}^{j-1} \operatorname{Exp}\left(\left(\tilde{\boldsymbol{\omega}}_{k}-\mathbf{b}_{i}^{g}\right) \Delta t-\mathbf{\eta}_{k}^{g d} \Delta t\right) \\
& \stackrel{1}\approx \prod_{k=i}^{j-1}\left\{\operatorname{Exp}\left(\left(\tilde{\boldsymbol{\omega}}_{k}-\mathbf{b}_{i}^{g}\right) \Delta t\right) \cdot \operatorname{Exp}\left(-\mathbf{J}_{r}\left(\left(\tilde{\boldsymbol{\omega}}_{k}-\mathbf{b}_{i}^{g}\right) \Delta t\right) \cdot \mathbf{\eta}_{k}^{g d} \Delta t\right)\right\} \\
& \stackrel{2}{=} \Delta \tilde{\mathbf{R}}_{i j} \cdot \prod_{k=i}^{j-1} \operatorname{Exp}\left(-\Delta \tilde{\mathbf{R}}_{k+1 j}^{T} \cdot \mathbf{J}_{r}^{k} \cdot \boldsymbol{\eta}_{k}^{g d} \Delta t\right)
\end{aligned}\tag{7}
$$
其中1处使用了性质：当$\delta \vec{\phi}$是小量时：
$$
\operatorname{Exp}(\vec{\phi}+\delta \vec{\phi}) \approx \operatorname{Exp}(\vec{\phi}) \cdot \operatorname{Exp}\left(\mathbf{J}_{r}(\vec{\phi}) \cdot \delta \vec{\phi}\right)\tag{8}
$$
其中2处利用Adjoint性质：
$$
\operatorname{Exp}(\vec{\phi}) \cdot \mathbf{R}=\mathbf{R} \cdot \operatorname{Exp}\left(\mathbf{R}^{T} \vec{\phi}\right)\tag{9}
$$
<font color=Red>FIXME:这里Adjoint性质性质是如何应用的？在式(7)中$R$是左乘的，而Adjoint性质是右乘，感觉不对呢？</font>

令
$$
\mathbf{J}_{r}^{k}=\mathbf{J}_{r}\left(\left(\tilde{\boldsymbol{\omega}}_{k}-\mathbf{b}_{i}^{g}\right) \Delta t\right)\tag{10}
$$
$$
\Delta \tilde{\mathbf{R}}_{i j}=\prod_{k=i}^{j-1} \operatorname{Exp}\left(\left(\tilde{\boldsymbol{\omega}}_{k}-\mathbf{b}_{i}^{g}\right) \Delta t\right)\tag{11}
$$
$$
\operatorname{Exp}\left(-\delta \vec{\phi}_{i j}\right)=\prod_{k=i}^{j-1} \operatorname{Exp}\left(-\Delta \tilde{\mathbf{R}}_{k+1 j}^{T} \cdot \mathbf{J}_{r}^{k} \cdot \mathbf{\eta}_{k}^{g d} \Delta t\right)\tag{12}
$$
则有：
$$
\Delta \mathbf{R}_{i j} \triangleq \Delta \tilde{\mathbf{R}}_{i j} \cdot \operatorname{Exp}\left(-\delta \vec{\phi}_{i j}\right)\tag{13}
$$
$\Delta \tilde{\mathbf{R}}_{i j}$即姿态预积分测量值，它由陀螺仪测量值和对陀螺仪偏差的估计得到，而$\delta \vec{\phi}_{i j}$或$\operatorname{Exp}\left(\delta \vec{\phi}_{i j}\right)$即测量噪声。
#### 3.2 $\Delta \mathbf{v}_{i j}$
将式（13）代入式（5），得到：
$$
\begin{aligned}
\Delta \mathbf{v}_{i j} &=\sum_{k=i}^{j-1} \Delta \mathbf{R}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}-\mathbf{\eta}_{k}^{a d}\right) \cdot \Delta t \\
& \approx \sum_{k=i}^{j-1} \Delta \tilde{\mathbf{R}}_{i k} \cdot \operatorname{Exp}\left(-\delta \vec{\phi}_{i k}\right) \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}-\mathbf{\eta}_{k}^{a d}\right) \cdot \Delta t \\
& \stackrel{1}\approx \sum_{k=i}^{j-1} \Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\mathbf{I}-\delta \vec{\phi}_{i k}^{\wedge}\right) \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}-\boldsymbol{\eta}_{k}^{a d}\right) \cdot \Delta t \\
& \stackrel{2}\approx \sum_{k=i}^{j-1}\left[\Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\mathbf{I}-\delta \vec{\phi}_{i k}^{\wedge}\right) \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right) \cdot \Delta t-\Delta \tilde{\mathbf{R}}_{i k} \mathbf{\eta}_{k}^{a d} \Delta t\right] \\
&\stackrel{3}=\sum_{k=i}^{j-1}\left[\Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right) \cdot \Delta t+\Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right)^{\wedge} \cdot \delta \vec{\phi}_{i k} \cdot \Delta t-\Delta \tilde{\mathbf{R}}_{i k} \mathbf{\eta}_{k}^{a d} \Delta t\right] \\
&=\sum_{k=i}^{j-1}\left[\Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right) \cdot \Delta t\right] \\
&+\sum_{k=i}^{j-1}\left[\Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right)^{\wedge} \cdot \delta \vec{\phi}_{i k} \cdot \Delta t-\Delta \tilde{\mathbf{R}}_{i k} \mathbf{\eta}_{k}^{a d} \Delta t\right]
\end{aligned}\tag{14}
$$
其中1处使用了“当$\vec{\phi}$是小量时，有一阶近似：$\exp \left(\vec{\phi}^{\wedge}\right) \approx \mathbf{I}+\vec{\phi}^{\wedge}$，或$\operatorname{Exp}(\vec{\phi}) \approx \mathbf{I}+\vec{\phi}^{\wedge}$”的性质。
其中2处忽略高阶小项$\delta \vec{\phi}_{i k}^{\wedge} \mathbf{\eta}_{k}^{a d}$。
其中3处使用了$\mathbf{a}^{\wedge} \cdot \mathbf{b}=-\mathbf{b}^{\wedge} \cdot \mathbf{a}$的性质。
再令：
$$
\begin{aligned}
\Delta \tilde{\mathbf{v}}_{i j} & \triangleq \sum_{k=i}^{j-1}\left[\Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right) \cdot \Delta t\right] \\
\delta \mathbf{v}_{i j} & \triangleq \sum_{k=i}^{j-1}\left[\Delta \tilde{\mathbf{R}}_{i k} \mathbf{\eta}_{k}^{a d} \Delta t-\Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right)^{\wedge} \cdot \delta \vec{\phi}_{i k} \cdot \Delta t\right]
\end{aligned}\tag{15}
$$
得到：
$$
\Delta \mathbf{v}_{i j} \triangleq \Delta \tilde{\mathbf{v}}_{i j}-\delta \mathbf{v}_{i j}\tag{16}
$$
$\tilde{\mathbf{v}}_{i j}$即速度增量预积分测量值，它由IMU测量值和对偏差的估计或猜测计算得到。$\delta \mathbf{v}_{i j}$即其测量噪声。

#### 3.3  $\Delta \mathbf{p}_{i j}$
将式（13）和（16）代入式（6）得到：
$$
\begin{aligned}
\Delta \mathbf{p}_{i j} &=\sum_{k=i}^{j-1}\left[\Delta \mathbf{v}_{i k} \cdot \Delta t+\frac{1}{2} \Delta \mathbf{R}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}-\boldsymbol{\eta}_{k}^{a d}\right) \cdot \Delta t^{2}\right] \\
& \approx \sum_{k=i}^{j-1}\left[\left(\Delta \tilde{\mathbf{v}}_{i k}-\delta \mathbf{v}_{i k}\right) \cdot \Delta t+\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \cdot \operatorname{Exp}\left(-\delta \vec{\phi}_{i k}\right) \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}-\boldsymbol{\eta}_{k}^{a d}\right) \cdot \Delta t^{2}\right] \\
& \stackrel{(1)}\approx \sum_{k=i}^{j-1}\left[\left(\Delta \tilde{\mathbf{v}}_{i k}-\delta \mathbf{v}_{i k}\right) \cdot \Delta t+\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\mathbf{I}-\delta \vec{\phi}_{i k}^{\wedge}\right) \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}-\boldsymbol{\eta}_{k}^{a d}\right) \cdot \Delta t^{2}\right] \\
& \stackrel{(2)}\approx \sum_{k=i}^{j-1}\left[\left(\Delta \tilde{\mathbf{v}}_{i k}-\delta \mathbf{v}_{i k}\right) \cdot \Delta t\right.\\
&\left.\qquad+\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\mathbf{I}-\delta \vec{\phi}_{i k}^{\wedge}\right) \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right) \cdot \Delta t^{2}-\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \boldsymbol{\eta}_{k}^{a d} \Delta t^{2}\right] \\
&\stackrel{(3)}{=} \sum_{k=i}^{j-1}\left[\Delta \tilde{\mathbf{v}}_{i k} \Delta t+\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right) \Delta t^{2}\right.\\
&\left.+\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right)^{\wedge} \delta \vec{\phi}_{i k} \Delta t^{2}-\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \boldsymbol{\eta}_{k}^{a d} \Delta t^{2}-\delta \mathbf{v}_{i k} \Delta t\right]
\end{aligned}\tag{17}
$$
其中(1)处使用了“当$\vec{\phi}$是小量时，有一阶近似：$\operatorname{Exp}(\vec{\phi}) \approx \mathbf{I}+\vec{\phi}^{\wedge}$”的性质。
其中(2)处忽略高阶小项$\delta \vec{\phi}_{i k}^{\wedge} \mathbf{\eta}_{k}^{a d}$
其中(3)处使用了$\mathbf{a}^{\wedge} \cdot \mathbf{b}=-\mathbf{b}^{\wedge} \cdot \mathbf{a}$的性质。
再令：
$$
\begin{aligned}
&\Delta \tilde{\mathbf{p}}_{i j} \triangleq \sum_{k=i}^{j-1}\left[\Delta \tilde{\mathbf{v}}_{i k} \Delta t+\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right) \Delta t^{2}\right] \\
&\delta \mathbf{p}_{i j} \triangleq \sum_{k=i}^{j-1}\left[\delta \mathbf{v}_{i k} \Delta t-\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right)^{\wedge} \delta \vec{\phi}_{i k} \Delta t^{2}+\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \boldsymbol{\eta}_{k}^{a d} \Delta t^{2}\right]
\end{aligned}\tag{18}
$$
得到：
$$
\Delta \mathbf{p}_{i j} \triangleq \Delta \tilde{\mathbf{p}}_{i j}-\delta \mathbf{p}_{i j}\tag{19}
$$
$\Delta \tilde{\mathbf{p}}_{i j}$即位置增量预积分测量值，它由IMU测量值和对偏差的估计得到。$\delta \mathbf{p}_{i j}$即其测量噪声。
#### 3.4  
上面得到预积分真值和测量值的关系如下：
$$
\begin{aligned}
&\Delta \mathbf{R}_{i j} \triangleq \Delta \tilde{\mathbf{R}}_{i j} \cdot \operatorname{Exp}\left(-\delta \vec{\phi}_{i j}\right) \\
&\Delta \mathbf{v}_{i j} \triangleq \Delta \tilde{\mathbf{v}}_{i j}-\delta \mathbf{v}_{i j} \\
&\Delta \mathbf{p}_{i j} \triangleq \Delta \tilde{\mathbf{p}}_{i j}-\delta \mathbf{p}_{i j}
\end{aligned}\tag{20}
$$
代入预积分真值表达式（4）（5）（6）得到：
$$
\begin{aligned}
&\Delta \tilde{\mathbf{R}}_{i j} \approx \Delta \mathbf{R}_{i j} \operatorname{Exp}\left(\delta \vec{\phi}_{i j}\right)=\mathbf{R}_{i}^{T} \mathbf{R}_{j} \operatorname{Exp}\left(\delta \vec{\phi}_{i j}\right) \\
&\Delta \tilde{\mathbf{v}}_{i j} \approx \Delta \mathbf{v}_{i j}+\delta \mathbf{v}_{i j}=\mathbf{R}_{i}^{T}\left(\mathbf{v}_{j}-\mathbf{v}_{i}-\mathbf{g} \cdot \Delta t_{i j}\right)+\delta \mathbf{v}_{i j} \\
&\Delta \tilde{\mathbf{p}}_{i j} \approx \Delta \mathbf{p}_{i j}+\delta \mathbf{p}_{i j}=\mathbf{R}_{i}^{T}\left(\mathbf{p}_{j}-\mathbf{p}_{i}-\mathbf{v}_{i} \cdot \Delta t_{i j}-\frac{1}{2} \mathbf{g} \cdot \Delta t_{i j}^{2}\right)+\delta \mathbf{p}_{i j}
\end{aligned}\tag{21}
$$
上述表达式即为预积分测量值（含IMU测量值及偏差估计值）与真值之间的关系，即形如“测量值=真值+噪声”的形式。

### 4、预积分测量噪声的分布形式

下面对预积分测量噪声进行分析，证明其符合高斯分布（目的是给出其协方差的计算表达式），令预积分的测量噪声为：
$$
\mathbf{\eta}_{i j}^{\Delta} \triangleq\left[\begin{array}{lll}
\delta \vec{\phi}_{i j}^{T} & \delta \mathbf{v}_{i j}^{T} & \delta \mathbf{p}_{i j}^{T}
\end{array}\right]^{T}\tag{22}
$$
我们希望其满足高斯分布，即$\boldsymbol{\eta}_{i j}^{\Delta} \sim N\left(\mathbf{0}_{9 \times 1}, \boldsymbol{\Sigma}_{i j}\right)$。由于$\boldsymbol{\eta}_{i j}^{\Delta}$是$\delta \vec{\phi}_{i j}^{T}$、$\delta \mathbf{v}_{i j}^{T}$、$\delta \mathbf{p}_{i j}^{T}$的线性组合，下面分别分析这三个噪声项的分布形式。
#### 4.1 $\delta \vec{\phi}_{i j}^{T}$的分布形式
对式12两边取对数有：
$$
\delta \vec{\phi}_{i j}=-\log \left(\prod_{k=i}^{j-1} \operatorname{Exp}\left(-\Delta \tilde{\mathbf{R}}_{k+1 j}^{T} \cdot \mathbf{J}_{r}^{k} \cdot \mathbf{\eta}_{k}^{g d} \Delta t\right)\right)\tag{23}
$$
令
$$
\boldsymbol{\xi}_{k}=\Delta \tilde{\mathbf{R}}_{k+1 j}^{T} \cdot \mathbf{J}_{r}^{k} \cdot \mathbf{\eta}_{k}^{g d} \Delta t\tag{24}
$$
由于$\mathbf{\eta}_{k}^{g d}$是小量，因此$\boldsymbol{\xi}_{k}$也是小量，于是$\mathbf{J}_{r}\left(\xi_{k}\right) \approx \mathbf{I}$，$\mathbf{J}_{r}^{-1}\left(\xi_{k}\right) \approx \mathbf{I}$，并利用BCH公式的近似形式
$$
\log (\operatorname{Exp}(\vec{\phi}) \cdot \operatorname{Exp}(\delta \vec{\phi}))=\vec{\phi}+\mathbf{J}_{r}^{-1}(\vec{\phi}) \cdot \delta \vec{\phi}\tag{25}
$$
对式（23）推导如下：
$$
\begin{aligned}
\delta \vec{\phi}_{i j} &=-\log \left(\prod_{k=i}^{j-1} \operatorname{Exp}\left(-\xi_{k}\right)\right) \\
&=-\log \left(\operatorname{Exp}\left(-\xi_{i}\right) \prod_{k=i+1}^{j-1} \operatorname{Exp}\left(-\xi_{k}\right)\right) \\
& \approx-\left(-\xi_{i}+\mathbf{I} \cdot \log \left(\prod_{k=i+1}^{j-1} \operatorname{Exp}\left(-\xi_{k}\right)\right)\right)=\xi_{i}-\log \left(\prod_{k=i+1}^{j-1} \operatorname{Exp}\left(-\xi_{k}\right)\right) \\
&=\xi_{i}-\log \left(\operatorname{Exp}\left(-\xi_{i+1}\right) \prod_{k=i+2}^{j-1} \operatorname{Exp}\left(-\xi_{k}\right)\right) \\
& \approx \xi_{i}+\xi_{i+1}-\log \left(\prod_{k=i+2}^{j-1} \operatorname{Exp}\left(-\xi_{k}\right)\right) \\
& \approx \cdots \\
& \approx \sum_{k=i}^{j-1} \xi_{k}
\end{aligned}\tag{26}
$$
即：
$$
\delta \vec{\phi}_{i j} \approx \sum_{k=i}^{j-1} \Delta \tilde{\mathbf{R}}_{k+1 j}^{T} \mathbf{J}_{r}^{k} \mathbf{\eta}_{k}^{g d} \Delta t\tag{27}
$$
由于$ \Delta \tilde{\mathbf{R}}_{k+1 j}^{T}$、$\mathbf{J}_{r}^{k}$和$ \Delta t$都是已知量，而$\mathbf{\eta}_{k}^{g d}$是零均值高斯噪声，因此$\delta \vec{\phi}_{i j}$(的一阶近似)也是零均值高斯噪声。
#### 4.2 $\delta \mathbf{v}_{i j}^{T}$的分布形式
由于$\delta \vec{\phi}_{i j}$近似拥有了高斯噪声的形式，且$\boldsymbol{\eta}_{k}^{a d}$也是零均值高斯噪声，根据$\delta \mathbf{v}_{i j}^{T}$的表达式可知其也拥有高斯分布的形式。

#### 4.3 $\delta \mathbf{p}_{i j}^{T}$的分布形式
类似于$\delta \mathbf{v}_{i j}^{T}$，$\delta \mathbf{p}_{i j}^{T}$也拥有高斯分布的形式。

### 5、预积分测量噪声的递推形式
下面推导预积分测量噪声的递推形式，即$\mathbf{\eta}_{i j-1}^{\Delta} \rightarrow \mathbf{\eta}_{i j}^{\Delta}$，及其协方差$\boldsymbol{\Sigma}_{i j}$的递推形式$\boldsymbol{\Sigma}_{i j-1} \rightarrow \boldsymbol{\Sigma}_{i j}$，下面依次推导$\delta \vec{\phi}_{i j-1} \rightarrow \delta \vec{\phi}_{i j}$、$\delta \mathbf{v}_{i j-1} \rightarrow \delta \mathbf{v}_{i j}$、$\delta \mathbf{p}_{i j-1} \rightarrow \delta \mathbf{p}_{i j}$。

#### 5.1、$\delta \vec{\phi}_{i j-1} \rightarrow \delta \vec{\phi}_{i j}$
$$
\begin{aligned}
\delta \vec{\phi}_{i j} &=\sum_{k=i}^{j-1} \Delta \tilde{\mathbf{R}}_{k+1 j}^{T} \mathbf{J}_{r}^{k} \mathbf{\eta}_{k}^{g d} \Delta t \\
&=\sum_{k=i}^{j-2} \Delta \tilde{\mathbf{R}}_{k+1 j}^{T} \mathbf{J}_{r}^{k} \boldsymbol{\eta}_{k}^{\operatorname{gd}} \Delta t+\Delta \tilde{\mathbf{R}}_{j j}^{T} \mathbf{J}_{r}^{j-1} \boldsymbol{\eta}_{j-1}^{g d} \Delta t \\
& \stackrel{1}{=} \sum_{k=i}^{j-2}\left(\Delta \tilde{\mathbf{R}}_{k+1 j-1} \Delta \tilde{\mathbf{R}}_{j-1 j}\right)^{T} \mathbf{J}_{r}^{k} \boldsymbol{\eta}_{k}^{g d} \Delta t+\mathbf{J}_{r}^{j-1} \boldsymbol{\eta}_{j-1}^{g d} \Delta t \\
&=\Delta \tilde{\mathbf{R}}_{j j-1} \sum_{k=i}^{j-2} \Delta \tilde{\mathbf{R}}_{k+1 j-1}^{T} \mathbf{J}_{r}^{k} \boldsymbol{\eta}_{k}^{g d} \Delta t+\mathbf{J}_{r}^{j-1} \boldsymbol{\eta}_{j-1}^{g d} \Delta t \\
&=\Delta \tilde{\mathbf{R}}_{j j-1} \delta \vec{\phi}_{i j-1}+\mathbf{J}_{r}^{j-1} \boldsymbol{\eta}_{j-1}^{g d} \Delta t
\end{aligned}\tag{28}
$$
其中1处利用了$\Delta \tilde{\mathbf{R}}_{j j}^{T}=\mathbf{I}$以及$\Delta \tilde{\mathbf{R}}_{l m} \Delta \tilde{\mathbf{R}}_{m n}=\Delta \tilde{\mathbf{R}}_{l n}$的性质，推导过程中进行了一些变形。

#### 5.2、$\delta \mathbf{v}_{i j-1} \rightarrow \delta \mathbf{v}_{i j}$

$$
\begin{aligned}
\delta \mathbf{v}_{i j}=& \sum_{k=i}^{j-1}\left[\Delta \tilde{\mathbf{R}}_{i k} \boldsymbol{\eta}_{k}^{\alpha d} \Delta t-\Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right)^{\wedge} \cdot \delta \vec{\phi}_{i k} \cdot \Delta t\right] \\
=& \sum_{k=i}^{j-2}\left[\Delta \tilde{\mathbf{R}}_{i k} \boldsymbol{\eta}_{k}^{a d} \Delta t-\Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right)^{\wedge} \cdot \delta \vec{\phi}_{i k} \cdot \Delta t\right] \ldots \\
&+\Delta \tilde{\mathbf{R}}_{i j-1} \boldsymbol{\eta}_{j-1}^{a d} \Delta t-\Delta \tilde{\mathbf{R}}_{i j-1} \cdot\left(\tilde{\mathbf{f}}_{j-1}-\mathbf{b}_{i}^{a}\right)^{\wedge} \cdot \delta \vec{\phi}_{i j-1} \cdot \Delta t \\
=& \delta \mathbf{v}_{i j-1}+\Delta \tilde{\mathbf{R}}_{i j-1} \boldsymbol{\eta}_{j-1}^{a d} \Delta t-\Delta \tilde{\mathbf{R}}_{i j-1} \cdot\left(\tilde{\mathbf{f}}_{j-1}-\mathbf{b}_{i}^{a}\right)^{\wedge} \cdot \delta \vec{\phi}_{i j-1} \cdot \Delta t
\end{aligned}\tag{29}
$$
直接进行加项拆分即可完成推导。

#### 5.3、$\delta \mathbf{p}_{i j-1} \rightarrow \delta \mathbf{p}_{i j}$
$$
\begin{aligned}
\delta \mathbf{p}_{i j}=& \sum_{k=i}^{j-1}\left[\delta \mathbf{v}_{i k} \Delta t-\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\mathbf{b}_{i}^{a}\right)^{\wedge} \delta \vec{\phi}_{i k} \Delta t^{2}+\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i k} \boldsymbol{\eta}_{k}^{a d} \Delta t^{2}\right] \\
=& \delta \mathbf{p}_{i j-1}+\delta \mathbf{v}_{i j-1} \Delta t \\
&-\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i j-1} \cdot\left(\tilde{\mathbf{f}}_{j-1}-\mathbf{b}_{i}^{a}\right)^{\wedge} \delta \vec{\phi}_{i j-1} \Delta t^{2}+\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i j-1} \boldsymbol{\eta}_{j-1}^{a d} \Delta t^{2}
\end{aligned}\tag{30}
$$
同样直接进行加项拆分即可完成推导。
#### 5.4、递推形式
令：
$$
\boldsymbol{\eta}_{k}^{d}=\left[\left(\boldsymbol{\eta}_{k}^{g d}\right)^{T} \quad\left(\mathbf{\eta}_{k}^{a d}\right)^{T}\right]^{T}\tag{31}
$$
综上可得$\boldsymbol{\eta}_{i j}^{\Delta}$的递推形式如下：
$$
\begin{aligned}
\boldsymbol{\eta}_{i j}^{\Delta}=&\left[\begin{array}{ccc}
\Delta \tilde{\mathbf{R}}_{j j-1} & \mathbf{0} & \mathbf{0} \\
-\Delta \tilde{\mathbf{R}}_{i j-1} \cdot\left(\tilde{\mathbf{f}}_{j-1}-\mathbf{b}_{i}^{a}\right)^{\wedge} \Delta t & \mathbf{I} & \mathbf{0} \\
-\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i j-1} \cdot\left(\tilde{\mathbf{f}}_{j-1}-\mathbf{b}_{i}^{a}\right)^{\wedge} \Delta t^{2} & \Delta t \mathbf{I} & \mathbf{I}
\end{array}\right] \mathbf{\eta}_{i j-1}^{\Delta} \ldots \\
&+\left[\begin{array}{cc}
\mathbf{J}_{r}^{j-1} \Delta t & \mathbf{0} \\
\mathbf{0} & \Delta \tilde{\mathbf{R}}_{i j-1} \Delta t \\
\mathbf{0} & \frac{1}{2} \Delta \tilde{\mathbf{R}}_{i j-1} \Delta t^{2}
\end{array}\right] \boldsymbol{\eta}_{j-1}^{d}
\end{aligned}\tag{32}
$$
令：
$$
\begin{aligned}
&\mathbf{A}_{j-1}=\left[\begin{array}{ccc}
\Delta \tilde{\mathbf{R}}_{j j-1} & \mathbf{0} & \mathbf{0} \\
-\Delta \tilde{\mathbf{R}}_{i j-1} \cdot\left(\tilde{\mathbf{f}}_{j-1}-\mathbf{b}_{i}^{a}\right)^{\wedge} \Delta t & \mathbf{I} & \mathbf{0} \\
-\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i j-1} \cdot\left(\tilde{\mathbf{f}}_{j-1}-\mathbf{b}_{i}^{a}\right)^{\wedge} \Delta t^{2} & \Delta t \mathbf{I} & \mathbf{I}
\end{array}\right] \\
&\mathbf{B}_{j-1}=\left[\begin{array}{cc}
\mathbf{J}_{r}^{j-1} \Delta t & \mathbf{0} \\
\mathbf{0} & \Delta \tilde{\mathbf{R}}_{i j-1} \Delta t \\
\mathbf{0} & \frac{1}{2} \Delta \tilde{\mathbf{R}}_{i j-1} \Delta t^{2}
\end{array}\right]
\end{aligned}\tag{33}
$$
则有IMU预积分测量噪声的递推形式：
$$
\mathbf{\eta}_{i j}^{\Delta}=\mathbf{A}_{j-1} \mathbf{\eta}_{i j-1}^{\Delta}+\mathbf{B}_{j-1} \mathbf{\eta}_{j-1}^{d}\tag{34}
$$
IMU预积分测量噪声的协方差矩阵就有了如下的递推计算形式：
$$
\boldsymbol{\Sigma}_{i j}=\mathbf{A}_{j-1} \boldsymbol{\Sigma}_{i j-1} \mathbf{A}_{j-1}^{T}+\mathbf{B}_{j-1} \mathbf{\Sigma}_{\boldsymbol{\eta}} \mathbf{B}_{j-1}^{T}\tag{35}
$$
<font color=Red>FIXME：从推导过程来看$\Sigma_{\eta}$应该是第j-1帧的测量噪声，但是在LIO-Livox的代码实现中，直接使用了一个12×12的常数噪声协方差矩阵noise来代替$\Sigma_{\eta}$，如下所示，且每一次递推都使用相同的值，这样合理吗？</font>
```
covariance = A * covariance * A.transpose() + B * noise * B.transpose();
```
<font color=Blue>从形式上看，IMU预积分协方差的递推形式类似于卡尔曼滤波中的状态变量协方差的预测方程，其中的$\mathbf{Q}$就相当于$\Sigma_{\eta}$，在每个递推周期都固定的加上这样一个常量噪声，表示从当前状态转移到下一个状态的过程中，存在各种噪声，总是会引入新的误差：</font>
$$
\overline{\mathbf{P}}=\mathbf{F P F}^{\top}+\mathbf{Q}
$$
IMU预积分测量噪声的协方差矩阵（即噪声分布）将用来计算信息矩阵，在优化框架中起到平衡权重的作用。在LIO-Livox中信息矩阵是这样构造的：
1. 求协方差矩阵的逆，相当于取了方差的倒数，方差越大，权重越小，反之，权重越大
2. 对逆矩阵进行Cholesky分解，获得下三角矩阵，相当于是求平方根，对权重进行衰减
3. 求下三角矩阵的转置
```
Eigen::Matrix <double, 15, 15> sqrt_information = Eigen::LLT <Eigen::Matrix<double, 15, 15>>(frame_curr->imuIntegrator.GetCovariance().inverse()).matrixL().transpose()
```
然后在IMU预积分代价函数的残差上左乘信息矩阵，如下所示：
```
eResiduals.applyOnTheLeft(sqrt_information.template cast<T>());
```
残差左乘信息矩阵能够起到平衡权重的作用：
1. 所谓信息矩阵是指IMU预积分测量噪声协方差矩阵的逆矩阵的平方根
2. 协方差矩阵的逆矩阵相当于取了方差的倒数，方差越大，权重越小，反之权重越大
3. 逆矩阵的平方根通过对协方差矩阵进行Cholesky分解获得，对权重进行衰减
4. 优化过程中误差只是减少并不是完全消除，不能消除的误差去哪里呢？当然是每条边（因子图）分摊了，但是每条边都分摊一样多的误差显然是不科学的，这个时候就需要信息矩阵，它表达了每条边要分摊的误差比例。


### 6、偏差更新时的预积分测量值更新

前面的预积分计算，都是在假设积分区间内陀螺和加计的偏差恒定的基础上推导的。当 bias 发生变化时，若仍按照前述公式，预积分测量值需要整个重新计算一遍，这将非常的耗费算力。为了解决这个问题，提出了利用线性化来进行偏差变化时预积分项的一阶近似更新方法。
 
 下面先给出各更新公式，首先做几个符号说明：$\overline{\mathbf{b}}_{i}^{g}$和$\overline{\mathbf{b}}_{i}^{a}$是旧的偏差，新的偏差$\hat{\mathbf{b}}_{i}^{g}$和$\hat{\mathbf{b}}_{a}^{a}$由旧偏差和更新量$\delta \mathbf{b}_{i}^{g}$和$\delta \mathbf{b}_{i}^{a}$相加得到：即$\hat{\mathbf{b}}_{i}^{g} \leftarrow \overline{\mathbf{b}}_{i}^{g}+\delta \mathbf{b}_{i}^{g}$、$\hat{\mathbf{b}}_{i}^{a} \leftarrow \overline{\mathbf{b}}_{i}^{a}+\delta \mathbf{b}_{i}^{a}$。
 于是有预积分关于偏差估计值变化的一阶近似更新公式如下：
$$
\begin{aligned}
&\Delta \tilde{\mathbf{R}}_{i j}\left(\hat{\mathbf{b}}_{i}^{g}\right) \approx \Delta \tilde{\mathbf{R}}_{i j}\left(\overline{\mathbf{b}}_{i}^{g}\right) \cdot \operatorname{Exp}\left(\frac{\partial \Delta \overline{\mathbf{R}}_{i j}}{\partial \overline{\mathbf{b}}^{g}} \delta \mathbf{b}_{i}^{g}\right) \\
&\Delta \tilde{\mathbf{v}}_{i j}\left(\hat{\mathbf{b}}_{i}^{g}, \hat{\mathbf{b}}_{i}^{a}\right) \approx \Delta \tilde{\mathbf{v}}_{i j}\left(\overline{\mathbf{b}}_{i}^{g}, \overline{\mathbf{b}}_{i}^{a}\right)+\frac{\partial \Delta \overline{\mathbf{v}}_{i j}}{\partial \overline{\mathbf{b}}^{g}} \delta \mathbf{b}_{i}^{g}+\frac{\partial \Delta \overline{\mathbf{v}}_{i j}}{\partial \overline{\mathbf{b}}^{a}} \delta \mathbf{b}_{i}^{a} \\
&\Delta \tilde{\mathbf{p}}_{i j}\left(\hat{\mathbf{b}}_{i}^{g}, \hat{\mathbf{b}}_{i}^{a}\right) \approx \Delta \tilde{\mathbf{p}}_{i j}\left(\overline{\mathbf{b}}_{i}^{g}, \overline{\mathbf{b}}_{i}^{a}\right)+\frac{\partial \Delta \overline{\mathbf{p}}_{i j}}{\partial \overline{\mathbf{b}}^{g}} \delta \mathbf{b}_{i}^{g}+\frac{\partial \Delta \overline{\mathbf{p}}_{i j}}{\partial \overline{\mathbf{b}}^{a}} \delta \mathbf{b}_{i}^{a}
\end{aligned}
$$
为了便于理解，做符号简化如下：
$$
\begin{aligned}
&\Delta \hat{\mathbf{R}}_{i j} \doteq \Delta \tilde{\mathbf{R}}_{i j}\left(\hat{\mathbf{b}}_{i}^{g}\right),  \&nbsp\&nbsp\&nbsp\&nbsp\&nbsp\&nbsp      \Delta \overline{\mathbf{R}}_{i j} \doteq \Delta \tilde{\mathbf{R}}_{i j}\left(\overline{\mathbf{b}}_{i}^{g}\right) \\
&\Delta \hat{\mathbf{v}}_{i j} \doteq \Delta \tilde{\mathbf{v}}_{i j}\left(\hat{\mathbf{b}}_{i}^{g}, \hat{\mathbf{b}}_{i}^{a}\right), \&nbsp\&nbsp\&nbsp\&nbsp       \Delta \overline{\mathbf{v}}_{i j} \doteq \Delta \tilde{\mathbf{v}}_{i j}\left(\overline{\mathbf{b}}_{i}^{g}, \overline{\mathbf{b}}_{i}^{a}\right) \\
&\Delta \hat{\mathbf{p}}_{i j} \doteq \Delta \tilde{\mathbf{p}}_{i j}\left(\hat{\mathbf{b}}_{i}^{g}, \hat{\mathbf{b}}_{i}^{a}\right), \&nbsp\&nbsp\&nbsp\&nbsp       \Delta \overline{\mathbf{p}}_{i j} \doteq \Delta \tilde{\mathbf{p}}_{i j}\left(\overline{\mathbf{b}}_{i}^{g}, \overline{\mathbf{b}}_{i}^{a}\right)
\end{aligned}
$$
得到简化后的公式如下：
$$
\begin{aligned}
&\Delta \hat{\mathbf{R}}_{i j} \approx \Delta \overline{\mathbf{R}}_{i j} \cdot \operatorname{Exp}\left(\frac{\partial \Delta \overline{\mathbf{R}}_{i j}}{\partial \overline{\mathbf{b}}^{g}} \delta \mathbf{b}_{i}^{g}\right) \\
&\Delta \hat{\mathbf{v}}_{i j} \approx \Delta \overline{\mathbf{v}}_{i j}+\frac{\partial \Delta \overline{\mathbf{v}}_{i j}}{\partial \overline{\mathbf{b}}^{g}} \delta \mathbf{b}_{i}^{g}+\frac{\partial \Delta \overline{\mathbf{v}}_{i j}}{\partial \overline{\mathbf{b}}^{a}} \delta \mathbf{b}_{i}^{a} \\
&\Delta \hat{\mathbf{p}}_{i j} \approx \Delta \overline{\mathbf{p}}_{i j}+\frac{\partial \Delta \overline{\mathbf{p}}_{i j}}{\partial \overline{\mathbf{b}}^{g}} \delta \mathbf{b}_{i}^{g}+\frac{\partial \Delta \overline{\mathbf{p}}_{i j}}{\partial \overline{\mathbf{b}}^{a}} \delta \mathbf{b}_{i}^{a}
\end{aligned}\tag{35}
$$
<font color=Blue>式（35）说明了IMU预积分是如何计算出测量值的修正值的，为什么雅可比能够起到修正值的作用？
1. 其中的$\Delta \overline{\mathbf{R}}_{i j},\Delta \overline{\mathbf{v}}_{i j},\Delta \overline{\mathbf{p}}_{i j}$表示旧的测量值，其中包含了旧的偏差$\overline{\mathbf{b}}_{i}^{g}$和$\overline{\mathbf{b}}_{i}^{a}$。
2. 其中的$\Delta \hat{\mathbf{R}}_{i j},\Delta \hat{\mathbf{v}}_{i j},\Delta \hat{\mathbf{p}}_{i j}$表示新的测量值，其中包含了 新的偏差$\hat{\mathbf{b}}_{i}^{g}$和$\hat{\mathbf{b}}_{a}^{a}$。
3. 新偏差=旧偏差+更新量$\delta \mathbf{b}_{i}^{g}$和$\delta \mathbf{b}_{i}^{a}$，那么，如果把测量值当做偏差的函数，只需要在旧的测量值上添加一个近似的修正量就可以获得近似的新测量值，而不需要重新积分。
4. 而这个修正量（增量）就是用偏差的更新量$\delta \mathbf{b}_{i}^{g}$和$\delta \mathbf{b}_{i}^{a}$乘以函数的导数（即斜率）获得。

<font color=Blue>这样一来，对于i、j两帧之间的IMU积分我们只需要做一次就可以了，通过旧测量值函数对偏差的导数（即雅可比）和偏差更新量$\delta \mathbf{b}_{i}^{g},\delta \mathbf{b}_{i}^{a}$就可以近似的计算出修正量，获得新测量值的近似值，而不需要重新积分。
如果优化过程中起始位姿发生了变化，则雅可比也相应更新。而偏差更新量$\delta \mathbf{b}_{i}^{g},\delta \mathbf{b}_{i}^{a}$本身就是待优化的变量之一，自然也是相应更新。从而测量值的修正量实现了自动更新。

<font color=Blue>以上就是IMU预积分避免重新积分，降低运算量的关键。

其中的偏导项定义如下（推导过程参见《IMU预积分总结与公式推导》）：
$$
\begin{aligned}
&\frac{\partial \Delta \overline{\mathbf{R}}_{i j}}{\partial \overline{\mathbf{b}}^{g}}=\sum_{k=i}^{j-1}\left(-\Delta \overline{\mathbf{R}}_{k+1 j}^{T} \mathbf{J}_{r}^{k} \Delta t\right) \\
&\frac{\partial \Delta \overline{\mathbf{v}}_{i j}}{\partial \overline{\mathbf{b}}^{g}}=-\sum_{k=i}^{j-1}\left(\Delta \overline{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\overline{\mathbf{b}}_{i}^{a}\right)^{\wedge} \frac{\partial \Delta \overline{\mathbf{R}}_{i k}}{\partial \overline{\mathbf{b}}^{g}} \Delta t\right) \\
&\frac{\partial \Delta \overline{\mathbf{v}}_{i j}}{\partial \overline{\mathbf{b}}^{a}}=-\sum_{k=i}^{j-1}\left(\Delta \overline{\mathbf{R}}_{i k} \Delta t\right) \\
&\frac{\partial \Delta \overline{\mathbf{p}}_{i j}}{\partial \overline{\mathbf{b}}^{g}}=\sum_{k=i}^{j-1}\left[\frac{\partial \Delta \overline{\mathbf{v}}_{i k}}{\partial \overline{\mathbf{b}}^{g}} \Delta t-\frac{1}{2} \Delta \overline{\mathbf{R}}_{i k} \cdot\left(\tilde{\mathbf{f}}_{k}-\overline{\mathbf{b}}_{i}^{a}\right)^{\wedge} \frac{\partial \Delta \overline{\mathbf{R}}_{i k}}{\partial \overline{\mathbf{b}}^{g}} \Delta t^{2}\right] \\
&\frac{\partial \Delta \overline{\mathbf{p}}_{i j}}{\partial \overline{\mathbf{b}}^{a}}=\sum_{k=i}^{j-1}\left[\frac{\partial \Delta \overline{\mathbf{v}}_{i k}}{\partial \overline{\mathbf{b}}^{a}} \Delta t-\frac{1}{2} \Delta \overline{\mathbf{R}}_{i k} \Delta t^{2}\right]
\end{aligned}\tag{36}
$$
其中$\mathbf{J}_{r}^{k}=\mathbf{J}_{r}\left(\left(\tilde{\boldsymbol{\omega}}_{k}-\mathbf{b}_{i}^{g}\right) \Delta t\right)$

### 7、优化与残差

在实际应用中，通常以$\mathbf{R}_{i}, \mathbf{p}_{i}, \mathbf{v}_{i}, \mathbf{R}_{j}, \mathbf{p}_{j}, \mathbf{v}_{j}$等为导航求解的目标，同时由于IMU的偏差也是不可忽视的，因此，全部的导航状态是$\mathbf{R}_{i}, \mathbf{p}_{i}, \mathbf{v}_{i}, \mathbf{R}_{j}, \mathbf{p}_{j}, \mathbf{v}_{j}, \delta \mathbf{b}_{i}^{g}, \delta \mathbf{b}_{i}^{a}$。

<font color=Blue>残差定义如下，其中第一部分是预积分的估计值（需要通过非IMU的方式获得），第二部分是预积分的测量值，前面推导的“**偏差更新时的预积分测量值更新**”在残差中进行了应用，这种近似的修正方式免去了积分的重新计算，是预积分降低计算量的关键。<font>
$$
\begin{aligned}
\mathbf{r}_{\Delta R_{i j}} & \triangleq \log \left\{\left[\tilde{\mathbf{R}}_{i j}\left(\overline{\mathbf{b}}_{i}^{g}\right) \cdot \operatorname{Exp}\left(\frac{\partial \Delta \overline{\mathbf{R}}_{i j}}{\partial \overline{\mathbf{b}}^{g}} \delta \mathbf{b}_{i}^{g}\right)\right]^{T} \cdot \mathbf{R}_{i}^{T} \mathbf{R}_{j}\right\} \\
& \triangleq \log \left[\left(\Delta \hat{\mathbf{R}}_{i j}\right)^{T} \Delta \mathbf{R}_{i j}\right] \\
\mathbf{r}_{\Delta v_{i j}} & \triangleq \mathbf{R}_{i}^{T}\left(\mathbf{v}_{j}-\mathbf{v}_{i}-\mathbf{g} \cdot \Delta t_{i j}\right)-\left[\Delta \tilde{\mathbf{v}}_{i j}\left(\overline{\mathbf{b}}_{i}^{g}, \overline{\mathbf{b}}_{i}^{a}\right)+\frac{\partial \Delta \overline{\mathbf{v}}_{i j}}{\partial \overline{\mathbf{b}}^{g}} \delta \mathbf{b}_{i}^{g}+\frac{\partial \Delta \overline{\mathbf{v}}_{i j}}{\partial \overline{\mathbf{b}}^{a}} \delta \mathbf{b}_{i}^{a}\right] \\
& \triangleq \Delta \mathbf{v}_{i j}-\Delta \hat{\mathbf{v}}_{i j} \\
\mathbf{r}_{\Delta \mathbf{p}_{i j}} & \triangleq \mathbf{R}_{i}^{T}\left(\mathbf{p}_{j}-\mathbf{p}_{i}-\mathbf{v}_{i} \cdot \Delta t_{i j}-\frac{1}{2} \mathbf{g} \cdot \Delta t_{i j}^{2}\right)-\left[\Delta \tilde{\mathbf{p}}_{i j}\left(\overline{\mathbf{b}}_{i}^{g}, \overline{\mathbf{b}}_{i}^{a}\right)+\frac{\partial \Delta \overline{\mathbf{p}}_{i j}}{\partial \overline{\mathbf{b}}^{g}} \delta \mathbf{b}_{i}^{g}+\frac{\partial \Delta \overline{\mathbf{p}}_{i j}}{\partial \overline{\mathbf{b}}^{a}} \delta \mathbf{b}_{i}^{a}\right] \\
& \triangleq \Delta \mathbf{p}_{i j}-\Delta \hat{\mathbf{p}}_{i j}
\end{aligned}\tag{37}
$$
有了残差，接下来就变成了一个非线性最小二乘问题，通过迭代求解增量的方式，不断更新状态变量，使损失函数下降：
$$
\begin{aligned}
&\mathbf{R}_{i} \leftarrow \mathbf{R}_{i} \cdot \operatorname{Exp}\left(\delta \vec{\phi}_{i}\right) \\ 
&\mathbf{p}_{i} \leftarrow \mathbf{p}_{i}+\mathbf{R}_{i} \cdot \delta \mathbf{p}_{i}\\
&\mathbf{v}_{i} \leftarrow \mathbf{v}_{i}+\delta \mathbf{v}_{i} \\
&\mathbf{R}_{j} \leftarrow \mathbf{R}_{j} \cdot \operatorname{Exp}\left(\delta \vec{\phi}_{j}\right)\\
&\mathbf{p}_{j} \leftarrow \mathbf{p}_{j}+\mathbf{R}_{j} \cdot \delta \mathbf{p}_{j}\\
&\mathbf{v}_{j} \leftarrow \mathbf{v}_{j}+\delta \mathbf{v}_{j} \\
&\delta \mathbf{b}_{i}^{g} \leftarrow \delta \mathbf{b}_{i}^{g}+\delta \mathbf{b}_{i}^{g}\\
&\delta \mathbf{b}_{i}^{a} \leftarrow \delta \mathbf{b}_{i}^{a}+\widetilde{\delta \mathbf{b}_{i}^{a}}
\end{aligned}\tag{38}
$$
在利用各类方法进行非线性最小二乘计算时，需要提供残差关于这些状态变量的 Jacobian。对于 姿态来说，一般更习惯采用扰动模型（详见《视觉SLAM 十四讲》P75，这种模型比直接对 李代数求导能获得更好的 Jacobian 形式），因此为了统一状态的表述形式，我们一般采用对 扰动/摄动/增量进行求导来获取 Jacobian 矩阵。

关于雅可比矩阵的推导，详见《IMU预积分总结与公式推导》。

### 8、LIO-Livox的实现
LIO-Livox构造的A、B矩阵如下：
$$
\begin{aligned}
&\mathbf{A}_{j-1}=\left[\begin{array}{ccc}
\mathbf{I} & -\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i j-1} \cdot\left(\tilde{\mathbf{f}}_{j-1}-\mathbf{b}_{i}^{a}\right)^{\wedge} \Delta t^{2} & \Delta t \mathbf{I} & \mathbf{0} & -\frac{1}{2} \Delta \tilde{\mathbf{R}}_{i j-1} \Delta t^{2}\\
\mathbf{0} & \Delta \tilde{\mathbf{R}}_{j j-1} & \mathbf{0} &-\mathbf{J}_{r}^{j-1} \Delta t  & \mathbf{0}\\
\mathbf{0} & -\Delta \tilde{\mathbf{R}}_{i j-1} \cdot\left(\tilde{\mathbf{f}}_{j-1}-\mathbf{b}_{i}^{a}\right)^{\wedge} \Delta t & \mathbf{I} & \mathbf{0} & -\Delta \tilde{\mathbf{R}}_{i j-1} \Delta t\\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I} & \mathbf{0} & \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I} 
\end{array}\right] \\
&\mathbf{B}_{j-1}=\left[\begin{array}{cc}
\mathbf{0} & \frac{1}{2} \Delta \tilde{\mathbf{R}}_{i j-1} \Delta t^{2} & \mathbf{0} & \mathbf{0} \\
\mathbf{J}_{r}^{j-1} \Delta t & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \Delta \tilde{\mathbf{R}}_{i j-1} \Delta t & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0}
\end{array}\right]
\end{aligned}\tag{39}
$$
看起来和式（33）中的A、B矩阵不一样，一方面顺序不一致，此外A矩阵的右侧多了两列。主要是原因是需要迭代更新的状态变量的顺序从R、v、p变成了p、R、v，此外，还多了两个：
$$
\left[\begin{array}{lll}
\mathbf{p}^{T} & \mathbf{R}^{T} & \mathbf{v}^{T} & {\delta \mathbf{b}_{i}^{g}}^{T} & {\delta \mathbf{b}_{i}^{a}}^{T}
\end{array}\right]^{T}
$$
这么做的目的是还可以用矩阵A实现偏差偏导项的递推更新，一举两得：
```
jacobian = A * jacobian;
```
具体的递推公式参见式（36）。