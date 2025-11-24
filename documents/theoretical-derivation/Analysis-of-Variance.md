# 方差分析（ANOVA）完整推导

## 一、单因素方差分析

### 1.1 符号说明
- $k$：组数（处理水平）
- $n_i$：第 $i$ 组的观测数
- $N = \sum_{i=1}^k n_i$：总观测数
- $y_{ij}$：第 $i$ 组的第 $j$ 个观测值
- $\bar{y}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} y_{ij}$：第 $i$ 组均值
- $\bar{y} = \frac{1}{N}\sum_{i=1}^k\sum_{j=1}^{n_i} y_{ij}$：总均值

### 1.2 平方和分解
$$y_{ij} - \bar{y} = (y_{ij} - \bar{y}_i) + (\bar{y}_i - \bar{y})$$

$$\text{SST} = \sum_{i=1}^k \sum_{j=1}^{n_i} (y_{ij} - \bar{y})^2 = \underbrace{\sum_{i=1}^k n_i(\bar{y}_i - \bar{y})^2}_{\text{SSA}} + \underbrace{\sum_{i=1}^k \sum_{j=1}^{n_i} (y_{ij} - \bar{y}_i)^2}_{\text{SSE}}$$

### 1.3 方差分析表
| 来源 | 平方和 | 自由度 | 均方 | F统计量 |
|------|--------|--------|------|----------|
| 组间 | SSA | $k-1$ | MSA = SSA/$(k-1)$ | $F = \frac{\text{MSA}}{\text{MSE}}$ |
| 误差 | SSE | $N-k$ | MSE = SSE/$(N-k)$ | |
| 总计 | SST | $N-1$ | | |

**自由度验证**: $(k-1) + (N-k) = N-1$ ✓

---

## 二、双因素方差分析（有交互作用）

### 2.1 符号说明
- $a$：因素A的水平数
- $b$：因素B的水平数
- $n$：每个单元格的观测数（均衡设计）
- $y_{ijk}$：在A的第 $i$ 水平、B的第 $j$ 水平的第 $k$ 个观测
- $\bar{y}_{ij\cdot}$：单元格均值
- $\bar{y}_{i\cdot\cdot}$：因素A的第 $i$ 水平均值
- $\bar{y}_{\cdot j\cdot}$：因素B的第 $j$ 水平均值
- $\bar{y}$：总均值

### 2.2 模型与分解
模型：$y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}$

离差分解：
$$y_{ijk} - \bar{y} = (\bar{y}_{i\cdot\cdot} - \bar{y}) + (\bar{y}_{\cdot j\cdot} - \bar{y}) + (\bar{y}_{ij\cdot} - \bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot j\cdot} + \bar{y}) + (y_{ijk} - \bar{y}_{ij\cdot})$$

### 2.3 平方和分解
$$\text{SST} = \underbrace{bn\sum_{i=1}^a (\bar{y}_{i\cdot\cdot} - \bar{y})^2}_{\text{SSA}} + \underbrace{an\sum_{j=1}^b (\bar{y}_{\cdot j\cdot} - \bar{y})^2}_{\text{SSB}} + \underbrace{n\sum_{i=1}^a\sum_{j=1}^b (\bar{y}_{ij\cdot} - \bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot j\cdot} + \bar{y})^2}_{\text{SSAB}} + \underbrace{\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^n (y_{ijk} - \bar{y}_{ij\cdot})^2}_{\text{SSE}}$$

### 2.4 方差分析表
| 来源 | 平方和 | 自由度 | 均方 | F统计量 |
|------|--------|--------|------|----------|
| 因素A | SSA | $a-1$ | MSA = SSA/$(a-1)$ | $F_A = \frac{\text{MSA}}{\text{MSE}}$ |
| 因素B | SSB | $b-1$ | MSB = SSB/$(b-1)$ | $F_B = \frac{\text{MSB}}{\text{MSE}}$ |
| 交互AB | SSAB | $(a-1)(b-1)$ | MSAB = SSAB/$[(a-1)(b-1)]$ | $F_{AB} = \frac{\text{MSAB}}{\text{MSE}}$ |
| 误差 | SSE | $ab(n-1)$ | MSE = SSE/$[ab(n-1)]$ | |
| 总计 | SST | $abn-1$ | | |

**自由度验证**: $(a-1) + (b-1) + (a-1)(b-1) + ab(n-1) = abn-1$ ✓

---

## 三、双因素方差分析（无交互作用）

### 3.1 模型与假设
模型：$y_{ijk} = \mu + \alpha_i + \beta_j + \varepsilon_{ijk}$

**前提条件**：通过有交互作用模型的检验，确认 $F_{AB}$ 不显著（即交互作用可忽略）后使用。

### 3.2 平方和分解
$$\text{SST} = \underbrace{bn\sum_{i=1}^a (\bar{y}_{i\cdot\cdot} - \bar{y})^2}_{\text{SSA}} + \underbrace{an\sum_{j=1}^b (\bar{y}_{\cdot j\cdot} - \bar{y})^2}_{\text{SSB}} + \underbrace{\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^n (y_{ijk} - \bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot j\cdot} + \bar{y})^2}_{\text{SSE}}$$

### 3.3 方差分析表
| 来源 | 平方和 | 自由度 | 均方 | F统计量 |
|------|--------|--------|------|----------|
| 因素A | SSA | $a-1$ | MSA = SSA/$(a-1)$ | $F_A = \frac{\text{MSA}}{\text{MSE}}$ |
| 因素B | SSB | $b-1$ | MSB = SSB/$(b-1)$ | $F_B = \frac{\text{MSB}}{\text{MSE}}$ |
| 误差 | SSE | $abn-a-b+1$ | MSE = SSE/$[abn-a-b+1]$ | |
| 总计 | SST | $abn-1$ | | |

**自由度验证**: $(a-1) + (b-1) + (abn-a-b+1) = abn-1$ ✓

**注意**：无交互作用模型的误差项SSE实际上是"有交互作用模型中的SSAB + SSE"合并而来，因此其自由度也相应合并为：$(a-1)(b-1) + ab(n-1) = abn-a-b+1$

---

## 四、分析流程总结

1. **单因素分析**：直接检验组间差异
2. **双因素分析**：
   - 首先进行**有交互作用**的完整分析
   - 若交互作用不显著，改用**无交互作用**的简化模型
   - 若交互作用显著，重点解释交互效应而非主效应