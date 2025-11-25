# 卡方检验推导

## 符号说明
- $O$: 观察频数（observed frequency）
- $E$: 期望频数（expected frequency）
- $k$: 类别数（number of categories）
- $n$: 总观测数（total observations）
- $r$: 行数（number of rows）
- $c$: 列数（number of columns）
- $\chi^2$: 卡方统计量（chi-square statistic）

---

## 拟合优度检验（Goodness-of-Fit Test）

### 目的
检验一个分类变量的观察分布是否与理论分布一致。

### 推导
- 零假设下，期望频数 $E_i = n p_i$，其中 $p_i$ 为理论概率
- 卡方统计量：
  $$
  \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
  $$
- $p_i$ 较小时，$n p_i (1-p_i) \approx E_i$，根据中心极限定理，标准化残差 $Z_i = \frac{O_i - E_i}{\sqrt{E_i}}$ 近似标准正态。
- 在零假设下，$\chi^2$ 近似服从自由度为 $k - 1$ 的卡方分布（自由度减少 due to $\sum O_i = n$）

---

## 独立性检验（Test of Independence）

### 目的
检验两个分类变量是否独立。

### 推导
- 列联表中，观察频数为 $O_{ij}$
- 在独立性零假设下，期望频数 $E_{ij} = \frac{R_i C_j}{n}$，其中 $R_i$ 为行和，$C_j$ 为列和
- 卡方统计量：
  $$
  \chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
  $$
- 在零假设下，$\chi^2$ 近似服从自由度为 $(r-1)(c-1)$ 的卡方分布（自由度减少 due to $\sum_i O_{ij} = C_j$, $\sum_j O_{ij} = R_i$）

---

## 备注
- 推导基于多项分布和正态近似，要求大样本（通常 $E_i \geq 5$）
- 卡方分布近似依赖于中心极限定理和二次型理论