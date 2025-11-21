# 硬间隔支持向量机（SVM）——公式推导

## 1. 符号说明
- 训练集：$\{(\mathbf{x}_i, y_i)\}_{i=1}^n$，其中 $\mathbf{x}_i \in \mathbb{R}^d$，$y_i \in \{-1, +1\}$
- 决策超平面：$\mathbf{w}^\top \mathbf{x} + b = 0$，其中 $\mathbf{w} \in \mathbb{R}^d$，$b \in \mathbb{R}$
- 函数间隔：$y_i(\mathbf{w}^\top \mathbf{x}_i + b)$
- 几何间隔：$\dfrac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|}$

目标：最大化几何间隔。

## 2. 原始优化问题
最大化几何间隔等价于：
$$
\max_{\mathbf{w}, b} \ \frac{1}{\|\mathbf{w}\|} \quad \text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \ \forall i
$$

等价转化为凸二次规划问题：
$$
\min_{\mathbf{w}, b} \ \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \ \forall i \tag{P}
$$

## 3. 拉格朗日对偶
引入拉格朗日乘子 $\alpha_i \geq 0$，构造拉格朗日函数：
$$
\mathcal{L}(\mathbf{w}, b, \alpha) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i \left[ y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 \right]
$$

对 $\mathbf{w}$ 和 $b$ 求偏导并令其为零：
- $\dfrac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \ \Rightarrow \ \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i \tag{1}$
- $\dfrac{\partial \mathcal{L}}{\partial b} = 0 \ \Rightarrow \ \sum_{i=1}^n \alpha_i y_i = 0 \tag{2}$

将 (1) 和 (2) 代入 $\mathcal{L}$，得对偶函数：
$$
\mathcal{L}_D(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j
$$

对偶优化问题为：
$$
\max_{\alpha} \ \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j \\
\text{subject to} \quad \alpha_i \geq 0 \ \forall i, \quad \sum_{i=1}^n \alpha_i y_i = 0 \tag{D}
$$

## 4. KKT 条件与支持向量
最优解满足 KKT 互补松弛条件：
$$
\alpha_i \left[ y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 \right] = 0, \quad \forall i
$$

- 若 $\alpha_i > 0$，则 $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$：$\mathbf{x}_i$ 为**支持向量**
- 若 $\alpha_i = 0$，该样本不影响决策边界

## 5. 决策函数
由 (1) 得分类器为：
$$
f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i^\top \mathbf{x} + b
$$

仅支持向量（$\alpha_i > 0$）参与求和。

任取一个支持向量 $j$，计算偏置项：
$$
b = y_j - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i^\top \mathbf{x}_j
$$